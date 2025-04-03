import datetime
import os

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
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

from data_manager import DataManager
from misc import Misc


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

    def fit(self, per, opt_model):
        dm = DataManager()
        stock_list = dm.load_stock_list()

        dict_df = {}
        ago = datetime.date.today() - relativedelta(months=4)

        # データを準備する
        for code in stock_list["code"]:
            df = dm.load_stock_data(code, start=ago.strftime("%Y-%m-%d"), end="end")
            dict_df[f"{code}"] = self.add_technical_indicators(df)

        windows = [30, 31, 32, 33, 34]
        model_names = []

        for window in windows:
            list_X, list_y = [], []

            for code in stock_list["code"]:
                df = dict_df[f"{code}"]
                if len(df) < window:
                    continue

                for i in range(len(df) - window):
                    # 説明変数
                    df_input = df.iloc[i : i + window]

                    # 目的変数
                    df_output = df.iloc[i + window : i + window + 1]

                    tmp_X, flag = self.prepare_input_data(df_input)
                    if not flag:
                        continue

                    standard_value = df_input.tail(1)["close"].values
                    if per > 1:
                        flag = df_output["close"].values >= standard_value * per
                    elif per <= 1:
                        flag = df_output["close"].values <= standard_value * per
                    tmp_y = 1 if flag[0] else 0

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
                verbose=0,
            )

            now = datetime.datetime.now()
            filename = now.strftime(f"model_%Y%m%d_%H%M%S_{window}.keras")

            dirname = f"{dm.base_dir}/model"
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            filename = os.path.join(dirname, filename)
            model.save(filename)
            model_names.append(
                [datetime.datetime.now().strftime("%Y-%m-%d"), filename, window]
            )

        model_names = pd.DataFrame(
            model_names, columns=["date", "model_name", "window"]
        )

        return model_names

    def predict(self, df_models):
        dm = DataManager()
        list_stocks = dm.load_stock_list()

        dict_df = {}
        ago = datetime.date.today() - relativedelta(months=4)

        # データを準備する
        for code in list_stocks["code"]:
            df = dm.load_stock_data(code, start=ago.strftime("%Y-%m-%d"), end="end")
            dict_df[f"{code}"] = self.add_technical_indicators(df)

        list_result = []

        for filename, window in zip(df_models["model_name"], df_models["window"]):
            model = load_model(os.path.join(f"{dm.base_dir}/model/", filename))

            for code, brand in zip(list_stocks["code"], list_stocks["brand"]):
                df = dict_df[f"{code}"]
                if len(df) < window:
                    continue

                # 説明変数
                df = df.tail(window)

                # 入力データの準備
                array_X, flag = self.prepare_input_data(df)
                if not flag:
                    continue

                # 予測する
                y_pred = model.predict(np.array([array_X]), verbose=0)

                list_result.append([code, brand, y_pred[0][0]])

        df_result = pd.DataFrame(list_result, columns=["code", "brand", "pred"])
        df_result = df_result.loc[df_result.groupby("code")["pred"].idxmax(), :]

        # 予測値が0.5以上の銘柄を抽出し、トップ50を取得する
        df_filtered = df_result[df_result["pred"] >= 0.5].copy()
        ex_num = 50
        df_extract = df_filtered.sort_values("pred", ascending=False).head(ex_num)

        nbd = Misc().get_next_business_day(datetime.date.today()).strftime("%Y-%m-%d")
        df_extract.loc[:, "date"] = nbd
        df_extract = df_extract[["date", "code", "brand", "pred"]]

        return df_extract
