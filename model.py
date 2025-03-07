import os
import pickle
from datetime import datetime

import numpy as np
from sklearn.utils import all_estimators
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import warnings

import pandas as pd
import numpy as np

pd.set_option("display.max_rows", None)


class ModelLibrary:
    def __init__(self):
        self.clf = None

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

    def add_labels(self, df_list):
        # 正解ラベルを作成する
        label_list = [self.check_price_change(df, 180) for df in df_list]

        return label_list

    def check_price_change(self, df, percentage, time_window=20):
        # ある時刻における株価を基準にして、そこからtime_window分以内にpercentage％上昇するか否かを判定する。
        # target_price = base_price + abs(base_price) * (percentage / 100)

        result = pd.DataFrame(np.zeros((len(df), 1)), columns=["Result"])

        for i in range(time_window):
            shifted_df = df.shift(-(i + 1))
            result[f"Result_{i + 1}"] = 0
            condition = (
                shifted_df["close"] > df["close"] * (1 + percentage / 100)
            ).values
            result.loc[condition, f"Result_{i + 1}"] = 1

        for i in range(time_window):
            result["Result"] += result[f"Result_{i + 1}"]
            result.drop(f"Result_{i + 1}", axis=1, inplace=True)

        result.loc[result["Result"] > 0, "Result"] = 1

        return result

    def prepare_training_data(self, df_list, label_list, window=10):
        # 学習データを準備する

        df_list = [df.dropna() for df in df_list]
        label_list = [label.dropna() for label in label_list]

        scaler = StandardScaler()

        X_list = []
        y_list = []

        for df, label in zip(df_list, label_list):
            array_X = np.array(df)
            array_y = np.array(label)

            for i in range(len(df) - window):
                tmp1 = scaler.fit_transform(array_X[i : i + window])
                tmp2 = array_y[i + window - 1]

                X_list.append(tmp1)
                y_list.append(tmp2)

        X_array = np.array(X_list)
        y_array = np.array(y_list)

        X_array_downsampled, y_array_downsampled = self.downsampling(X_array, y_array)

        return X_array_downsampled, y_array_downsampled

    def downsampling(self, X_array, y_array):
        # ラベル1のインデックスを取得
        label_1_indices = np.where(y_array == 1)[0]
        # ラベル0のインデックスを取得
        label_0_indices = np.where(y_array == 0)[0]

        # ラベル1のサンプル数と同じ数だけ、ラベル0のデータをランダムにサンプリング
        downsampled_label_0_indices = resample(
            label_0_indices,
            replace=False,
            n_samples=len(label_1_indices),
            random_state=42,
        )

        # ダウンサンプリングしたデータのインデックスとラベル1のインデックスを結合
        selected_indices = np.concatenate(
            [downsampled_label_0_indices, label_1_indices]
        )

        # X_arrayとy_arrayから選択したインデックスのデータを取得
        X_downsampled = X_array[selected_indices]
        y_downsampled = y_array[selected_indices]

        return X_downsampled, y_downsampled

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
