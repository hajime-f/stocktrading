import datetime
import sqlite3

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import EarlyStopping
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
        self.scaler = StandardScaler()

    def add_technical_indicators(self, df):
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

    def compile_lstm(self, shape1, shape2):
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

    def compile_rnn(self, shape1, shape2):
        model = Sequential()

        model.add(InputLayer(shape=(shape1, shape2)))
        model.add(Bidirectional(SimpleRNN(200)))
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

    def fit(self, per, opt_model, nbd):
        dm = DataManager()
        dm.set_token()

        df_stocks = pd.DataFrame(dm.fetch_stock_list())
        dict_df = {}
        ago = nbd - relativedelta(months=4)

        # データを準備する
        for _, tmp in df_stocks.iterrows():
            code = tmp["Code"][:-1]
            df = dm.load_stock_data(
                code, start=ago.strftime("%Y-%m-%d"), end=nbd.strftime("%Y-%m-%d")
            )
            if df.empty or df["date"].tail(1).item() != nbd.strftime("%Y-%m-%d"):
                continue
            df = self.add_technical_indicators(df)
            array_std = self.scaler.fit_transform(np.array(df))
            dict_df[f"{code}"] = pd.DataFrame(array_std)
            dict_df[f"{code}"] = pd.concat(
                [dict_df[f"{code}"], df["close"].reset_index(drop=True)], axis=1
            )

        window = 30
        list_X, list_y = [], []

        for df in dict_df.values():
            for i in range(len(df) - window):
                df_input = df.iloc[i : i + window]
                df_output = df.iloc[i + window : i + window + 1]
                list_X.append(df_input.drop(columns="close"))

                standard_value = df_input.tail(1)["close"].item()
                if per > 1:
                    flag = df_output["close"].item() >= standard_value * per
                elif per <= 1:
                    flag = df_output["close"].item() <= standard_value * per
                list_y.append(1 if flag else 0)

        array_X = np.array(list_X)
        array_y = np.array(list_y)

        # モデルの学習
        if opt_model == "lstm":
            model = self.compile_lstm(array_X.shape[1], array_X.shape[2])
        elif opt_model == "rnn":
            model = self.compile_rnn(array_X.shape[1], array_X.shape[2])
        else:
            raise ValueError(
                "不正なモデル名です。「lstm」または「rnn」を指定してください。"
            )
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

    def predict(self, model, nbd):
        dm = DataManager()
        dm.set_token()

        df_stocks = pd.DataFrame(dm.fetch_stock_list())
        dict_df = {}
        ago = nbd - relativedelta(months=4)
        list_com = []

        for _, tmp in df_stocks.iterrows():
            code = tmp["Code"][:-1]
            df = dm.load_stock_data(
                code, start=ago.strftime("%Y-%m-%d"), end=nbd.strftime("%Y-%m-%d")
            )
            if df.empty or df["date"].tail(1).item() != nbd.strftime("%Y-%m-%d"):
                continue
            df = self.add_technical_indicators(df)
            array_std = self.scaler.fit_transform(np.array(df))
            dict_df[f"{code}"] = pd.DataFrame(array_std)
            list_com.append([code, tmp["CompanyName"]])

        df_com = pd.DataFrame(list_com, columns=["code", "brand"])
        list_result = []
        window = 30

        for code, brand in zip(df_com["code"], df_com["brand"]):
            array_X = np.array(dict_df[f"{code}"].tail(window))
            y_pred = model.predict(np.array([array_X]), verbose=0)
            list_result.append([code, brand, y_pred[0][0]])

        df_result = pd.DataFrame(list_result, columns=["code", "brand", "pred"])
        df_extract = df_result[df_result["pred"] >= 0.8].copy()

        try:
            nbd = Misc().get_next_business_day(nbd)
            df_extract.loc[:, "date"] = nbd
            df_extract = df_extract[["date", "code", "brand", "pred"]]
            return df_extract
        except Exception:
            return pd.DataFrame()


if __name__ == "__main__":
    mm = ModelManager()
    dm = DataManager()
    misc = Misc()
    ave = pd.DataFrame()

    day = "2025-03-31"
    nbd = datetime.datetime.strptime(day, "%Y-%m-%d")

    while True:
        model = mm.fit(per=1.005, opt_model="lstm", nbd=nbd)
        df_long = mm.predict(model, nbd)
        df_long.loc[:, "side"] = 2

        model = mm.fit(per=0.995, opt_model="lstm", nbd=nbd)
        df_short = mm.predict(model, nbd)
        df_short.loc[:, "side"] = 1

        df = pd.concat([df_long, df_short])
        df = df.sort_values("pred", ascending=False).drop_duplicates(
            subset=["code"], keep="first"
        )

        selected_indices = []
        for index, row in df.iterrows():
            close_price = dm.find_newest_close_price(row["code"])
            if close_price < 10000:
                selected_indices.append(index)
        df = df.loc[selected_indices, :]

        weights = df["pred"].to_numpy()
        probabilities = weights / np.sum(weights)

        trial = 20
        df_stack = pd.DataFrame()

        for i in range(1, trial + 1):
            try:
                sampled_indices = np.random.choice(
                    a=df.index,
                    size=50,
                    replace=False,
                    p=probabilities,
                )
                df_tmp = df.loc[
                    sampled_indices, ["date", "code", "brand", "pred", "side"]
                ]
            except Exception:
                df_tmp = df.loc[:, ["date", "code", "brand", "pred", "side"]]

            df_tmp = df_tmp.sort_values("pred", ascending=False).reset_index(drop=True)
            df_tmp.loc[:, "trial"] = i

            df_stack = pd.concat([df_stack, df_tmp])

        df_stack = df_stack.reset_index(drop=True)

        total = []
        nbd_next = misc.get_next_business_day(nbd)

        for i in range(1, trial + 1):
            df_trial = df_stack.loc[df_stack["trial"] == i, :]
            change = []

            for index, row in df_trial.iterrows():
                prices = dm.load_open_close_prices(
                    row["code"], nbd_next.strftime("%Y-%m-%d")
                )

                try:
                    open_price = prices["open"].item()
                    close_price = prices["close"].item()
                except Exception:
                    continue

                if row["side"] == 1:
                    change.append(open_price - close_price)
                else:
                    change.append(close_price - open_price)

            total.append(int(sum(change) * 100))

        data = int(sum(total) / len(total))
        tmp = pd.DataFrame(
            data, columns=["average"], index=[nbd_next.strftime("%Y-%m-%d")]
        )
        ave = pd.concat([ave, tmp], axis=0)

        nbd = nbd_next
        if misc.get_next_business_day(nbd) >= datetime.date.today():
            break

        print(f"{nbd.strftime('%Y-%m-%d')} - {data} 円")

    print(ave)
    breakpoint()
