import os
import pickle
from datetime import datetime

import numpy as np
from sklearn.utils import all_estimators
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report
import warnings

from crawler import Crawler

import pandas as pd

pd.set_option("display.max_rows", None)


class ModelLibrary:
    def __init__(self, n_symbols):
        self.n_symbols = n_symbols
        self.data = []
        for _ in range(n_symbols):
            self.data.append([])
        self.clf = None

    def append_data(self, new_data, index):
        if new_data["CurrentPriceTime"] is not None:
            data = {
                "CurrentPriceTime": new_data["CurrentPriceTime"],
                "CurrentPrice": new_data["CurrentPrice"],
                "TradingVolume": new_data["TradingVolume"],
            }
            self.data[index].append(data)

    def save_data(self, data_list):
        now = datetime.now()
        filename = now.strftime("data_%Y%m%d_%H%M%S.pkl")

        dirname = "./data"
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        filename = os.path.join(dirname, filename)

        with open(filename, "wb") as f:
            pickle.dump(data_list, f)

        return filename

    def set_data(self, p_data):
        concat_data = []

        for d in p_data:
            concat_data += d

        self.data = concat_data

    def prepare_dataframe_list(self, symbols):
        # 生データを分足のDataFrameに変換する
        df_list = [self.convert_to_dataframe(d) for d in self.data]

        # データを正規化する
        df_list = [
            self.normalize_data(df, symbol) for df, symbol in zip(df_list, symbols)
        ]

        return df_list

    def add_technical_indicators(self, df_list):
        # 移動平均を計算する
        df_list = [self.calc_moving_average(df) for df in df_list]

        # MACDを計算する
        df_list = [self.calc_macd(df) for df in df_list]

        # ボリンジャーバンドを計算する
        df_list = [self.calc_bollinger_band(df) for df in df_list]

        # RSIを計算する
        df_list = [self.calc_rsi(df) for df in df_list]

        return df_list

    def add_labels(self, df_list):
        # 正解ラベルを作成する
        label_df_list = [self.check_price_change(df["close"], 180) for df in df_list]

        # データを結合する
        XY = [
            pd.concat(
                [input_df.reset_index(drop=True), label_df.reset_index(drop=True)],
                axis=1,
            ).dropna()
            for input_df, label_df in zip(df_list, label_df_list)
        ]

        return XY

    def convert_to_dataframe(self, original_data):
        price_list = []

        for d in original_data:
            if d["CurrentPriceTime"] is None:
                continue

            dt_object = datetime.fromisoformat(
                d["CurrentPriceTime"].replace("Z", "+00:00")
            )
            formatted_datetime = dt_object.strftime("%Y-%m-%d %H:%M")
            price_list.append([formatted_datetime, d["CurrentPrice"]])

        price_df = pd.DataFrame(price_list, columns=["DateTime", "Price"]).set_index(
            "DateTime"
        )
        price_df.index = pd.to_datetime(price_df.index)
        price_df = price_df.resample("1Min").ohlc().dropna()
        price_df.columns = price_df.columns.get_level_values(1)

        return price_df

    def normalize_data(self, data, symbol):
        crawler = Crawler(symbol)
        values = crawler.fetch_stock_data()
        max_value, min_value = crawler.extract_first_row_data(values)
        try:
            max_value, min_value = (
                float(max_value.replace(",", "")),
                float(min_value.replace(",", "")),
            )
        except ValueError:
            max_value, min_value = data.iloc[0]["high"], data.iloc[0]["low"]

        return (data - min_value) / (max_value - min_value)

    def calc_moving_average(self, data):
        data["MA5"] = data["close"].rolling(window=5).mean()
        data["MA25"] = data["close"].rolling(window=25).mean()

        return data

    def calc_macd(self, data):
        data["MACD"] = (
            data["close"].ewm(span=12).mean() - data["close"].ewm(span=26).mean()
        )
        data["SIGNAL"] = data["MACD"].ewm(span=9).mean()
        data["HISTOGRAM"] = data["MACD"] - data["SIGNAL"]

        return data

    def calc_bollinger_band(self, data):
        sma20 = data["close"].rolling(window=20).mean()
        std20 = data["close"].rolling(window=20).std()
        data["Upper"] = sma20 + (std20 * 2)
        data["Lower"] = sma20 - (std20 * 2)

        return data

    def calc_rsi(self, data):
        delta = data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data["RSI"] = (100 - (100 / (1 + rs))) / 100

        return data

    def check_price_change(self, stock_price, percentage, time_window=20):
        # ある時刻における株価を基準にして、そこからtime_window分以内にpercentage％変化するか否かを判定する。

        result = []

        for i, base_price in enumerate(stock_price):
            # 目標株価を計算
            target_price = base_price + abs(base_price) * (percentage / 100)

            start_index = i + 1
            end_index = start_index + time_window

            if end_index <= len(stock_price):
                # 基準時刻からtime_window分後の株価を取得
                future_prices = stock_price.iloc[start_index:end_index]

                if percentage > 0:
                    if (future_prices > target_price).any():
                        result.append(1)
                    else:
                        result.append(0)
                elif percentage < 0:
                    if (future_prices < target_price).any():
                        result.append(1)
                    else:
                        result.append(0)
                else:
                    result.append(0)

            else:
                result.append(np.nan)

        return pd.DataFrame(result, columns=["Result"])

    def balance_dataframe(self, df, target_column="Result"):
        # 値の出現回数をカウント
        counts = df[target_column].value_counts()

        if len(counts) == 1:
            return pd.DataFrame(columns=df.columns)

        # 少数派の数を取得
        minority_count = counts.min()

        # 各グループからランダムに minority_count 個のサンプルを抽出
        balanced_df = []
        for value in counts.index:
            group = df[df[target_column] == value]
            sampled_group = group.sample(n=minority_count, random_state=42)
            balanced_df.append(sampled_group)

        # 抽出されたサンプルを結合
        balanced_df = pd.concat(balanced_df)

        return balanced_df

    def prepare_training_data(self, raw_data, window=10):
        X = pd.DataFrame()
        Y = pd.DataFrame()

        for r in raw_data:
            for i in range(len(r) - window):
                tmp1 = r.drop(["Result"], axis=1).iloc[i : i + window]
                tmp2 = r.Result.iloc[i + window - 1]

                X = pd.concat([X, pd.DataFrame([tmp1.values.reshape(-1)])])
                Y = pd.concat([Y, pd.DataFrame([tmp2])])

        X = X.reset_index(drop=True)
        Y = Y.reset_index(drop=True)

        Y.columns = ["Result"]
        XY = self.balance_dataframe(pd.concat([X, Y], axis=1))

        X = XY.drop(["Result"], axis=1)
        Y = XY["Result"]

        return X, Y

    def evaluate_model(self, X, Y):
        # クロスバリデーション用のオブジェクトをインスタンス化する
        kfold_cv = KFold(n_splits=6, shuffle=False)
        warnings.filterwarnings("ignore")

        # classifier のアルゴリズムをすべて取得する
        all_Algorithms = all_estimators(type_filter="classifier")
        warnings.filterwarnings("ignore")

        best_clf = None
        max_score = -1

        # 各分類アルゴリズムをクロスバリデーションで評価する
        for name, algorithm in all_Algorithms:
            try:
                clf = algorithm()
                if hasattr(clf, "score"):
                    scores = cross_val_score(clf, X, Y, cv=kfold_cv)
                    m = round(np.mean(scores) * 100, 2)
                    print(name, "の正解率：", m, "％")
                    if max_score < m:
                        best_clf = clf
                        max_score = m

            except Exception:
                pass

        return best_clf

    def validate_model(self, clf, X, Y):
        # データを学習用データとテスト用データに分割する
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )

        # モデルを学習する
        clf = clf.fit(X_train, Y_train)

        # モデルを評価する
        Y_pred = clf.predict(X_test)
        print(classification_report(Y_test, Y_pred))

        return clf

    def save_model(self, model):
        now = datetime.now()
        filename = now.strftime("model_%Y%m%d_%H%M%S.pkl")

        dirname = "./model"
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        filename = os.path.join(dirname, filename)

        with open(filename, "wb") as f:
            pickle.dump(model, f)

        return filename

    def load_model(self, filename):
        with open(filename, "rb") as f:
            self.clf = pickle.load(f)

        return self.clf

    def predict(self, data):
        return self.clf.predict(data)
