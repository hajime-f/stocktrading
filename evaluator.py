import datetime

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


class Evaluator:
    def __init__(self):
        self.dm = DataManager()
        self.scaler = StandardScaler()

        stock_list = self.dm.load_stock_list()
        self.dict_df = {}

        # データを準備する
        for code in stock_list["code"]:
            df = self.dm.load_stock_data(code)
            df = self.add_technical_indicators(df)
            self.dict_df[f"{code}"] = self.normalize_data(df)

    def normalize_data(self, df):
        price_min = df["low"].min()
        price_max = df["high"].max()

        df["open"] = (df["open"] - price_min) / (price_max - price_min)
        df["high"] = (df["high"] - price_min) / (price_max - price_min)
        df["low"] = (df["low"] - price_min) / (price_max - price_min)
        df["close_save"] = df["close"]
        df["close"] = (df["close"] - price_min) / (price_max - price_min)
        df["MA5"] = (df["MA5"] - price_min) / (price_max - price_min)
        df["MA25"] = (df["MA25"] - price_min) / (price_max - price_min)
        df["Upper"] = (df["Upper"] - price_min) / (price_max - price_min)
        df["Lower"] = (df["Lower"] - price_min) / (price_max - price_min)

        volume_min = df["volume"].min()
        volume_max = df["volume"].max()
        df["volume"] = (df["volume"] - volume_min) / (volume_max - volume_min)

        df["MACD"] = df["MACD"] / 100
        df["SIGNAL"] = df["SIGNAL"] / 100
        df["HISTOGRAM"] = df["HISTOGRAM"] / 100
        df["RSI"] = df["RSI"] / 100

        return df

    def add_technical_indicators(self, df):
        # 日付をインデックスにする
        df.set_index("date", inplace=True)
        df.index = pd.to_datetime(df.index)

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

        # # 始値と終値の差を追加する
        # df["trunk"] = df["open"] - df["close"]

        # 移動平均線乖離率を追加する
        df["MA5_rate"] = (df["close"] - df["MA5"]) / df["MA5"]
        df["MA25_rate"] = (df["close"] - df["MA25"]) / df["MA25"]

        # MACDの乖離率を追加する
        df["MACD_rate"] = (df["MACD"] - df["SIGNAL"]) / df["SIGNAL"]

        # RSIの乖離率を追加する
        df["RSI_rate"] = (df["RSI"] - 50) / 50

        # ボリンジャーバンドの乖離率を追加する
        df["Upper_rate"] = (df["close"] - df["Upper"]) / df["Upper"]

        # # 移動平均の差を追加する
        # df["MA_diff"] = df["MA5"] - df["MA25"]

        # nan を削除
        df = df.dropna()

        return df

    def prepare_input_data(self, df):
        array = np.array(df)

        try:
            array_std = self.scaler.fit_transform(array)
        except ValueError:
            return None, False

        return np.array(array_std), True

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

    def fit(self, per, opt_model, dict_df_sub):
        window = 30
        list_X, list_y = [], []

        for df in dict_df_sub.values():
            for i in range(len(df) - window):
                df_input = df.iloc[i : i + window]
                df_output = df.iloc[i + window : i + window + 1]

                tmp_X = np.array(df_input.drop("close_save", axis=1))

                standard_value = df_input.tail(1)["close_save"].item()
                comp_value = df_output["close_save"].item()

                if per > 1:
                    flag = comp_value >= standard_value * per
                elif per <= 1:
                    flag = comp_value <= standard_value * per
                tmp_y = 1 if flag else 0

                list_X.append(tmp_X)
                list_y.append(tmp_y)

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
            # verbose=0,
        )

        return model

    def predict(self, model):
        list_result = []
        window = 30

        for code, df in self.dict_df.items():
            # 説明変数
            df_input = df.tail(window)

            # 入力データの準備
            array_X, flag = self.prepare_input_data(df_input)
            if not flag:
                continue

            # 予測する
            y_pred = model.predict(np.array([array_X]), verbose=0)

            list_result.append([code, y_pred[0][0]])

        df_result = pd.DataFrame(list_result, columns=["code", "pred"])

        # 予測値が0.5以上の銘柄を抽出し、トップ50を取得する
        df_filtered = df_result[df_result["pred"] >= 0.5].copy()
        ex_num = 50
        df_extract = df_filtered.sort_values("pred", ascending=False).head(ex_num)

        # nbd = datetime.date.today().strftime("%Y-%m-%d")
        nbd = Misc().get_next_business_day(datetime.date.today()).strftime("%Y-%m-%d")
        df_extract.loc[:, "date"] = nbd
        df_extract = df_extract[["date", "code", "brand", "pred"]]

        return df_extract


if __name__ == "__main__":
    evaluator = Evaluator()

    for i in range(50, 0, -1):
        std_day = datetime.date.today() - relativedelta(days=i)
        ago = std_day - relativedelta(months=3)

        std_day = std_day.strftime("%Y-%m-%d")
        ago = ago.strftime("%Y-%m-%d")

        dict_df_sub = {}

        for code, df in evaluator.dict_df.items():
            dict_df_sub[code] = df.loc[ago:std_day]

        print("=======")
        model = evaluator.fit(1.005, "lstm", dict_df_sub)
        print("=======\n\n")
