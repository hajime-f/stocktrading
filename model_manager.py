import datetime
import sqlite3

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import metrics
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    InputLayer,
    SimpleRNN,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from data_manager import DataManager
from misc import Misc

pd.set_option("display.max_rows", None)


class ModelManager:
    def __init__(self):
        self.dm = DataManager()
        self.window = 30

    def add_technical_indicators(self, df):
        df = df.copy()

        # 日付をインデックスにする
        df.set_index("date", inplace=True)

        # 移動平均線を追加する
        df["MA5"] = df["close"].rolling(window=5).mean()
        df["MA25"] = df["close"].rolling(window=25).mean()

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
        epsilon = 1e-10
        rs = gain / (loss + epsilon)
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

    def compile_model(self, shape1, shape2, rnn_layer):
        model = Sequential()
        model.add(InputLayer(shape=(shape1, shape2)))
        model.add(Bidirectional(rnn_layer))
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

    def prepare_stock_data(self, ratio=0.3):
        list_stocks = self.dm.load_stock_list()

        today = datetime.date.today()
        start_date = today - relativedelta(months=4)

        dict_data_for_model, dict_data_for_scaler = {}, {}
        for code in list_stocks["code"]:
            df = self.dm.load_stock_data(code, start=start_date)
            df = self.add_technical_indicators(df)
            n_data = int(len(df) * ratio)
            dict_data_for_model[f"{code}"] = df[n_data:]
            dict_data_for_scaler[f"{code}"] = df[:n_data]

        return dict_data_for_model, dict_data_for_scaler

    def train_scaler(self, dict_data):
        scaler = StandardScaler()
        scaler.fit(pd.concat(dict_data.values()))
        return scaler

    def fit(self, dict_data, scaler, per, opt_model="lstm", window=30, epochs=15):
        list_X, list_y = [], []

        for df in dict_data.values():
            scaled_values = scaler.transform(df)
            df_scaled = pd.DataFrame(scaled_values, index=df.index, columns=df.columns)

            for i in range(len(df) - window):
                list_X.append(df_scaled.iloc[i : i + window])

                current_close = df.iloc[i : i + window].tail(1)["close"].item()
                future_close = df.iloc[i + window : i + window + 1]["close"].item()

                if per > 1:
                    flag = future_close >= current_close * per
                else:
                    flag = future_close <= current_close * per
                list_y.append(1 if flag else 0)

        array_X = np.array(list_X)
        array_y = np.array(list_y)

        if opt_model == "lstm":
            layer = LSTM(200)
        elif opt_model == "rnn":
            layer = SimpleRNN(200)
        else:
            pass

        model = self.compile_model(array_X.shape[1], array_X.shape[2], layer)
        model.fit(array_X, array_y, batch_size=128, epochs=epochs, verbose=True)

        return model

    def predict(self, model, scaler, dict_data, window=30):
        list_result = []
        for code, df in dict_data.items():
            df_scaled = scaler.transform(df.tail(window))
            array_X = np.array(df_scaled)
            y_pred = model.predict(np.array([array_X]), verbose=0)
            list_result.append([code, self.dm.get_brand(code), y_pred[0][0]])

        df_result = pd.DataFrame(list_result, columns=["code", "brand", "pred"])
        df_extract = df_result[df_result["pred"] >= 0.5].copy()

        # nbd = datetime.date.today().strftime("%Y-%m-%d")
        nbd = Misc().get_next_business_day(datetime.date.today()).strftime("%Y-%m-%d")
        df_extract.loc[:, "date"] = nbd
        df_extract = df_extract[["date", "code", "brand", "pred"]]

        return df_extract


if __name__ == "__main__":
    mm = ModelManager()

    dict_data_for_model, dict_data_for_scaler = mm.prepare_stock_data()

    # スケーラーの学習
    scaler = mm.train_scaler(dict_data_for_scaler)

    # ショートモデルの学習・予測
    model = mm.fit(dict_data_for_model, scaler, per=0.995)
    df_short = mm.predict(model, scaler, dict_data_for_model)
    df_short.loc[:, "side"] = 1

    # ロングモデルの学習・予測
    model = mm.fit(dict_data_for_model, scaler, per=1.005)
    df_long = mm.predict(model, scaler, dict_data_for_model)
    df_long.loc[:, "side"] = 2

    df = pd.concat([df_long, df_short])
    df = df.sort_values("pred", ascending=False).drop_duplicates(
        subset=["code"], keep="first"
    )

    dm = DataManager()
    selected_indices = []

    for index, row in df.iterrows():
        close_price = dm.find_newest_close_price(row["code"])
        if close_price < 8000:
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

    df = df.loc[sampled_indices, ["date", "code", "brand", "pred", "side"]]
    df = df.sort_values("pred", ascending=False).reset_index()

    conn = sqlite3.connect(dm.db)
    with conn:
        df.to_sql("Target2", conn, if_exists="append", index=False)
