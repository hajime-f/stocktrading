import datetime
import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from config_manager import cm
from data_manager import DataManager
from library import Library
from misc import Misc


class ModelManager:
    def __init__(self):
        self.dm = DataManager()
        self.df_stock_list = self.dm.load_stock_list()

        self.lib = Library()

        self.threshold = float(cm.get("model.threshold"))
        self.window = int(cm.get("model.window_size"))
        self.det_per = float(cm.get("model.det_per"))

        t = datetime.date.today()
        self.nbd = Misc.get_next_business_day(t).strftime("%Y-%m-%d")

    def add_technical_indicators(self, df):
        # 日付をインデックスにする
        df.set_index("date", inplace=True)

        # 移動平均線を追加する
        df["MA5"] = df["close"].rolling(window=5).mean()
        df["MA25"] = df["close"].rolling(window=25).mean()
        df["volume_MA20"] = df["volume"].rolling(window=20).mean()

        # 対数変換する
        df["log_close"] = np.log(df["close"])

        # 差分を取る
        df["diff"] = df["close"].diff()

        # MACDを追加する
        df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
        df["SIGNAL"] = df["MACD"].ewm(span=9).mean()
        df["HISTOGRAM"] = df["MACD"] - df["SIGNAL"]

        # ボリンジャーバンドを追加する
        sma20 = df["close"].rolling(window=20).mean()
        std20 = df["close"].rolling(window=20).std()
        df["Upper"] = sma20 + (std20 * 2)
        df["Lower"] = sma20 - (std20 * 2)

        # RSIを追加する
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # 終値の前日比を追加する
        df_shift = df.shift(1)
        df["close_rate"] = (df["close"] - df_shift["close"]) / df_shift["close"]

        # 始値と終値の差を追加する
        df["trunk"] = df["open"] - df["close"]

        # 移動平均線乖離率を追加する
        df["MA5_rate"] = (df["close"] - df["MA5"]) / df["MA5"]
        df["MA25_rate"] = (df["close"] - df["MA25"]) / df["MA25"]

        # MACDの乖離率を追加する
        df["MACD_rate"] = (df["MACD"] - df["SIGNAL"]) / df["SIGNAL"]

        # RSIの乖離率を追加する
        df["RSI_rate"] = (df["RSI"] - 50) / 50

        # ボリンジャーバンドの乖離率を追加する
        df["Upper_rate"] = (df["close"] - df["Upper"]) / df["Upper"]

        # 移動平均の差を追加する
        df["MA_diff"] = df["MA5"] - df["MA25"]

        # nan を削除
        df = df.dropna()

        return df

    def compile_model(self, shape1, shape2):
        model = Sequential()
        model.add(InputLayer(shape=(shape1, shape2)))
        model.add(Bidirectional(LSTM(200)))
        model.add(Dropout(0.3))
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy", metrics.Precision(), metrics.Recall()],
        )

        return model

    def prepare_data(self):
        # データを準備するメソッド

        scaler = StandardScaler()

        dict_df = {}
        dict_df_close = {}

        today = datetime.date.today()
        start = (today - relativedelta(months=4)).strftime("%Y-%m-%d")

        for code in self.df_stock_list["code"]:
            df = self.dm.load_stock_data(code, start=start)
            df = self.add_technical_indicators(df)
            dict_df[code] = pd.DataFrame(scaler.fit_transform(df), index=df.index)
            dict_df_close[code] = df["close"]

        return dict_df, dict_df_close

    def evaluate_feature_importance(self, dict_df_learn, dict_df_close, per):
        # 特徴量の重要度を評価するメソッド
        # （学習・予測には使わないメソッドなので見なくてよい）

        list_X, list_y = [], []
        window = self.window

        original_cols = list(dict_df_learn.values())[0].columns

        for code in dict_df_learn.keys():
            df_scaled = dict_df_learn[code]
            df_close = dict_df_close[code]

            for i in range(len(df_scaled) - window):
                window_X = df_scaled.iloc[i : i + window]

                last_date_of_window = window_X.index[-1]
                loc = df_close.index.get_loc(last_date_of_window)

                current_close = df_close.iloc[loc]
                future_close = df_close.iloc[loc + 1]
                label = self.create_label(current_close, future_close, per)

                list_X.append(window_X.to_numpy())
                list_y.append(label)

        X = np.array(list_X)
        y = np.array(list_y)

        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape((n_samples, n_timesteps * n_features))

        xgb_model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
        )
        xgb_model.fit(X_reshaped, y)

        feature_names = [
            f"{col}_t-{i}" for i in range(window - 1, -1, -1) for col in original_cols
        ]

        importances = xgb_model.feature_importances_
        df_importance = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        )

        df_importance["original_feature"] = df_importance["feature"].apply(
            lambda x: x.rsplit("_t-", 1)[0]
        )
        df_agg_importance = (
            df_importance.groupby("original_feature")["importance"]
            .sum()
            .sort_values(ascending=False)
        )

        df_plot_data = df_agg_importance.head(50).sort_values(ascending=True)

        plt.style.use("ggplot")
        fig, ax = plt.subplots(figsize=(10, 12))
        ax.barh(y=df_plot_data.index, width=df_plot_data.values)

        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        plt.tight_layout()
        plt.show()

    def fit(self, dict_df, dict_df_close, per):
        # 予測器（LSTM）を学習させるメソッド

        list_X, list_y = [], []
        window = self.window

        for code in dict_df.keys():
            df = dict_df[code]
            df_close = dict_df_close[code]

            for i in range(len(df) - window):
                window_X = df.iloc[i : i + window]

                last_date_of_window = window_X.index[-1]
                loc = df_close.index.get_loc(last_date_of_window)

                current_close = df_close.iloc[loc]
                future_close = df_close.iloc[loc + 1]

                label = self.create_label(current_close, future_close, per)

                list_X.append(window_X)
                list_y.append(label)

        array_X, array_y = np.array(list_X), np.array(list_y)
        # array_X, array_y = shuffle(array_X, array_y, random_state=42)

        model = self.compile_model(array_X.shape[1], array_X.shape[2])
        model.fit(
            array_X,
            array_y,
            batch_size=128,
            epochs=30,
            validation_split=0.2,
            callbacks=[EarlyStopping(patience=3)],
            verbose=0,
        )

        return model

    def create_label(self, current_close, future_close, per):
        # ラベルを計算するメソッド

        if per > 1:
            flag = future_close >= current_close * per
        elif per <= 1:
            flag = future_close <= current_close * per
        return 1 if flag else 0

    def predict(self, model, dict_df, per):
        # 予測値を得るためのメソッド

        list_result = []

        for code in dict_df.keys():
            close_price = self.dm.find_newest_close_price(code)
            if not (700 < close_price < 5500):
                continue

            array_X = np.array(dict_df[code].tail(self.window))
            y_pred = model.predict(np.array([array_X]), verbose=0)

            df = self.df_stock_list
            brand = df[df["code"] == code]["brand"].iloc[0]

            list_result.append([code, brand, y_pred[0][0]])

        result = pd.DataFrame(list_result, columns=["code", "brand", "pred"])
        return result

    def select_candidate(self, df_long, df_short):
        # 予測結果に基づいて明日売買する銘柄を決定するメソッド

        df_long = df_long[df_long["pred"] >= self.threshold].copy()
        df_short = df_short[df_short["pred"] >= self.threshold].copy()

        df_long.loc[:, "side"] = 2  # 買い建て
        df_short.loc[:, "side"] = 1  # 売り建て

        df = pd.concat([df_long, df_short])
        df = df.sort_values("pred", ascending=False).drop_duplicates(
            subset=["code"], keep="first"
        )

        selected_indices = []
        for index, row in df.iterrows():
            # 信用売りにプレミアム料が乗る銘柄はスキップする
            if row["side"] == 1 and self.lib.examine_premium(row["code"]):
                continue
            # 売買制限がかかっている銘柄はスキップする
            if self.lib.examine_regulation(row["code"]):
                continue
            selected_indices.append(index)
        df = df.loc[selected_indices, :]

        # 予測値に応じて確率的に50銘柄を選抜する
        weights = df["pred"].to_numpy()
        probabilities = weights / np.sum(weights)
        sampled_indices = np.random.choice(
            a=df.index,
            size=50,
            replace=False,
            p=probabilities,
        )
        df = df.loc[sampled_indices, ["code", "brand", "pred", "side"]]
        df = df.sort_values("pred", ascending=False).reset_index()

        df.loc[:, "date"] = self.nbd
        df = df[["date", "code", "brand", "pred", "side"]]

        return df

    def save_result(self, df):
        conn = sqlite3.connect(self.dm.db)
        with conn:
            df.to_sql("Target", conn, if_exists="append", index=False)


if __name__ == "__main__":
    mm = ModelManager()

    # データの準備
    dict_df, dict_df_close = mm.prepare_data()

    # # 特徴量の重要度を評価
    # mm.evaluate_feature_importance(dict_df_learn, dict_df_close, 1.005)
    # breakpoint()

    # ロングモデルの学習
    long_model = mm.fit(dict_df, dict_df_close, 1 + mm.det_per)

    # ロングモデルの予測
    df_long = mm.predict(long_model, dict_df, 1 + mm.det_per)

    # ショートモデルの学習
    short_model = mm.fit(dict_df, dict_df_close, 1 - mm.det_per)

    # ショートモデルの予測
    df_short = mm.predict(short_model, dict_df, 1 - mm.det_per)

    # 最終候補を得る
    df = mm.select_candidate(df_long, df_short)

    # 結果を保存する
    mm.save_result(df)
