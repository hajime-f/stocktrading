import datetime
import sqlite3

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from data_manager import DataManager
from library import Library
from misc import Misc


class ModelManager:
    def __init__(self):
        self.dm = DataManager()
        self.df_stock_list = self.dm.load_stock_list()

        self.lib = Library()

        self.train_split_ratio = 0.8
        self.threshold = 0.5
        self.window = 30

    def add_technical_indicators(self, df):
        # 日付をインデックスにする
        df.set_index("date", inplace=True)

        # 移動平均線を追加する
        df["MA5"] = df["close"].rolling(window=5).mean()
        df["MA25"] = df["close"].rolling(window=25).mean()
        df["volume_MA20"] = df["volume"].rolling(window=20).mean()

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
        scaler = StandardScaler()

        dict_df_learn = {}  # 学習用
        dict_df_test = {}  # テスト用
        dict_df_close = {}  # 答え

        today = datetime.date.today()
        start = (today - relativedelta(months=6)).strftime("%Y-%m-%d")

        for code in self.df_stock_list["code"]:
            df = self.dm.load_stock_data(code, start=start)
            df = self.add_technical_indicators(df)

            df_learn = df.iloc[:-1]
            df_test = df.tail(self.window)
            dict_df_close[code] = df["close"]

            train_split_index = int(len(df_learn) * self.train_split_ratio)
            scaler.fit(df_learn.iloc[:train_split_index])

            dict_df_learn[code] = pd.DataFrame(
                scaler.transform(df_learn), index=df_learn.index
            )
            dict_df_test[code] = pd.DataFrame(
                scaler.transform(df_test), index=df_test.index
            )

        return dict_df_learn, dict_df_test, dict_df_close

    def fit(self, dict_df_learn, dict_df_close, per):
        list_X_train, list_y_train = [], []
        list_X_val, list_y_val = [], []
        window = self.window

        for code in dict_df_learn.keys():
            df_scaled = dict_df_learn[code]
            df_close = dict_df_close[code]

            X, y = [], []
            for i in range(len(df_scaled) - window + 1):
                window_X = df_scaled.iloc[i : i + window]

                last_date_of_window = window_X.index[-1]
                loc = df_close.index.get_loc(last_date_of_window)

                current_close = df_close.iloc[loc]
                future_close = df_close.iloc[loc + 1]
                label = self.create_label(current_close, future_close, per)

                X.append(window_X)
                y.append(label)

            split_index = int(len(X) * self.train_split_ratio)
            list_X_train.extend(X[:split_index])
            list_y_train.extend(y[:split_index])
            list_X_val.extend(X[split_index:])
            list_y_val.extend(y[split_index:])

        X_train, y_train = np.array(list_X_train), np.array(list_y_train)
        X_val, y_val = np.array(list_X_val), np.array(list_y_val)

        X_train, y_train = shuffle(X_train, y_train, random_state=42)

        model = self.compile_model(X_train.shape[1], X_train.shape[2])
        model.fit(
            X_train,
            y_train,
            batch_size=128,
            epochs=30,
            validation_data=(X_val, y_val),
            callbacks=[EarlyStopping(patience=3)],
        )

        return model

    def create_label(self, current_close, future_close, per):
        if per > 1:
            flag = future_close >= current_close * per
        elif per <= 1:
            flag = future_close <= current_close * per
        return 1 if flag else 0

    def predict(self, model, dict_df_test, per):
        list_result = []

        for code in dict_df_test.keys():
            close_price = self.dm.find_newest_close_price(code)
            if not (700 < close_price < 5500):
                continue

            array_X = np.array(dict_df_test[code])
            y_pred = model.predict(np.array([array_X]), verbose=0)

            df = self.df_stock_list
            brand = df[df["code"] == code]["brand"].iloc[0]

            list_result.append([code, brand, y_pred[0][0]])

        result = pd.DataFrame(list_result, columns=["code", "brand", "pred"])
        return result

    def get_candidate(self, df_long, df_short):
        df_long = df_long[df_long["pred"] >= self.threshold].copy()
        df_short = df_short[df_short["pred"] >= self.threshold].copy()

        df_long.loc[:, "side"] = 2
        df_short.loc[:, "side"] = 1

        df = pd.concat([df_long, df_short])
        df = df.sort_values("pred", ascending=False).drop_duplicates(
            subset=["code"], keep="first"
        )

        selected_indices = []
        for index, row in df.iterrows():
            if not self.lib.examine_regulation(row["code"]):
                selected_indices.append(index)
        df = df.loc[selected_indices, :]

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

        nbd = Misc.get_next_business_day(datetime.date.today()).strftime("%Y-%m-%d")
        df.loc[:, "date"] = nbd
        df = df[["date", "code", "brand", "pred", "side"]]

        return df

    def save_result(self, df):
        conn = sqlite3.connect(self.dm.db)
        with conn:
            df.to_sql("Target2", conn, if_exists="append", index=False)


if __name__ == "__main__":
    mm = ModelManager()

    # データの準備
    dict_df_learn, dict_df_test, dict_df_close = mm.prepare_data()

    # ロングモデルの学習
    long_model = mm.fit(dict_df_learn, dict_df_close, 1.005)

    # ロングモデルの予測
    df_long = mm.predict(long_model, dict_df_test, 1.005)

    # ショートモデルの学習
    short_model = mm.fit(dict_df_learn, dict_df_close, 0.995)

    # ショートモデルの予測
    df_short = mm.predict(short_model, dict_df_test, 0.995)

    # 最終候補を得る
    df = mm.get_candidate(df_long, df_short)

    # 結果を保存する
    mm.save_result(df)
