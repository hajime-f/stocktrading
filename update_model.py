import os
from datetime import datetime
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense, InputLayer, Dropout, SimpleRNN, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from data_manager import DataManager


class UpdateModel:
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

    def add_labels(self, df, percentage=0.5, day_window=1):
        result = pd.DataFrame(np.zeros((len(df), 1)), columns=["increase"])

        for i in range(day_window):
            shifted_df = df.shift(-(i + 1))
            result[f"increase_{i + 1}"] = 0
            condition = (
                shifted_df["close"] > df["close"] * (1 + percentage / 100)
            ).values
            result.loc[condition, f"increase_{i + 1}"] = 1

        result["increase"] = 0
        for i in range(day_window):
            result["increase"] += result[f"increase_{i + 1}"]
            result.drop(f"increase_{i + 1}", axis=1, inplace=True)

        result.loc[result["increase"] > 0, "increase"] = 1
        result.index = df.index

        df = df.iloc[:-day_window]
        result = result.iloc[:-day_window]

        return df, result

    def prepare_input_data(self, df):
        array = np.array(df)

        try:
            array_std = self.scaler.fit_transform(array)
        except ValueError:
            return None, False

        return np.array(array_std), True

    def compile_model(self, shape1, shape2):
        model = Sequential()

        model.add(InputLayer(shape=(shape1, shape2)))
        model.add(Bidirectional(SimpleRNN(200)))
        model.add(Dropout(0.3))
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation="sigmoid"))

        optimizer = Adam(learning_rate=0.001)

        model.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
        )

        return model


if __name__ == "__main__":
    dm = DataManager()
    stock_list = dm.load_stock_list()

    model = UpdateModel()
    model_names = []

    test_size = 30
    windows = [30, 31, 32, 33, 34]

    for window in windows:
        list_X = []
        list_y = []

        for code, brand in zip(stock_list["code"], stock_list["brand"]):
            df = dm.load_stock_data(code, start="2019-01-01", end="end")

            if window * test_size > len(df):
                continue

            # テクニカル指標を追加
            df = model.add_technical_indicators(df)

            # day_window日以内の終値が当日よりpercentage%以上上昇していたらフラグを立てる
            df, result = model.add_labels(df)

            for j in range(test_size, 0, -1):
                df_test = df.iloc[-window - j : -j]
                result_test = result.iloc[-window - j : -j]

                tmp_X, flag = model.prepare_input_data(df_test)
                if not flag:
                    continue
                tmp_y = result_test.tail(1).values

                list_X.append(tmp_X)
                list_y.append(tmp_y)

        array_X = np.array(list_X)
        array_y = np.array(list_y)

        # モデルの学習
        pred_model = model.compile_model(array_X.shape[1], array_X.shape[2])
        pred_model.fit(
            array_X,
            array_y,
            batch_size=128,
            epochs=30,
            validation_split=0.2,
            callbacks=[EarlyStopping(patience=3)],
            verbose=0,
        )

        # モデルの保存
        now = datetime.now()
        filename = now.strftime(f"model_swingtrade_%Y%m%d_%H%M%S_{window}.keras")

        dirname = "/Users/hajime-f/Development/stocktrading/model"
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        filename = os.path.join(dirname, filename)
        pred_model.save(filename)
        model_names.append([datetime.now().strftime("%Y-%m-%d"), filename, window])

    # モデルのファイル名をデータベースに保存する
    model_names = pd.DataFrame(model_names, columns=["date", "model_name", "window"])
    dm.save_model_names(model_names)
