import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, InputLayer
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from data_manager import DataManager


class Predictor:
    def __init__(self):
        self.dm = DataManager()
        self.df_stock_list = self.dm.load_stock_list()
        self.scaler = StandardScaler()

        self.window = 60
        self.split_ratio = 0.8

    def add_technical_indicators(self, df):
        # 日付をインデックスにする
        df.set_index("date", inplace=True)

        # 移動平均線を追加する
        df["MA5"] = df["close"].rolling(window=5).mean()
        df["MA25"] = df["close"].rolling(window=25).mean()

        # # 対数変換する
        # df["log_close"] = np.log(df["close"])

        # # 差分を取る
        # df["diff"] = df["close"].diff()

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

        # # 終値の前日比を追加する
        # df_shift = df.shift(1)
        # df["close_rate"] = (df["close"] - df_shift["close"]) / df_shift["close"]

        # 移動平均線乖離率を追加する
        df["MA5_rate"] = (df["close"] - df["MA5"]) / df["MA5"]
        df["MA25_rate"] = (df["close"] - df["MA25"]) / df["MA25"]

        # nan を削除
        df = df.dropna()

        return df

    def create_dataset(self, array_data, array_close):
        X, y = [], []

        for i in range(self.window, len(array_data)):
            X.append(array_data[i - self.window : i])
            y.append(array_close[i])
        return np.array(X), np.array(y)

    def prepare_data(self, code):
        df = self.dm.load_stock_data(code)
        df = self.add_technical_indicators(df)

        self.feature_names = df.columns.tolist()

        df = pd.DataFrame(
            self.scaler.fit_transform(df),
            index=df.index,
            columns=df.columns,
        )
        X, y = self.create_dataset(np.array(df), np.array(df["close"]))

        n_split1 = int(len(X) * self.split_ratio)
        n_split2 = int(n_split1 * self.split_ratio)

        array_learn_X, array_learn_y = X[:n_split2], y[:n_split2]
        array_val_X, array_val_y = X[n_split2:n_split1], y[n_split2:n_split1]
        array_test_X, array_test_y = X[n_split1:], y[n_split1:]
        array_pred_X = np.array(df.tail(self.window))[np.newaxis, :, :]

        return (
            array_learn_X,
            array_learn_y,
            array_val_X,
            array_val_y,
            array_test_X,
            array_test_y,
            array_pred_X,
        )

    def compile_model(self, shape1, shape2):
        model = Sequential()
        model.add(InputLayer(shape=(shape1, shape2)))
        model.add(Bidirectional(LSTM(200)))
        model.add(Dropout(0.3))
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation="linear"))

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="mean_squared_error",
            metrics=["mae", RootMeanSquaredError(name="rmse")],
        )

        return model

    def fit(self, array_learn_X, array_learn_y, array_val_X, array_val_y):
        model = self.compile_model(array_learn_X.shape[1], array_learn_X.shape[2])

        model.fit(
            array_learn_X,
            array_learn_y,
            batch_size=128,
            epochs=30,
            validation_data=(array_val_X, array_val_y),
            callbacks=[EarlyStopping(patience=3)],
            # verbose=0,
        )

        return model

    def predict(self, model, array_test_X):
        return model.predict(array_test_X, verbose=0)

    def evaluate_model(self, array_pred, array_test_y):
        return math.sqrt(mean_squared_error(array_test_y, array_pred))

    def draw_figure(self, pred, actual):
        plt.figure(figsize=(12, 7))
        plt.plot(actual, color="blue")
        plt.plot(pred, color="red")
        plt.show()

    def inverse_transform(self, array):
        dummy_array = np.zeros((len(array), len(self.feature_names)))
        target_col_index = self.feature_names.index("close")
        dummy_array[:, target_col_index] = array.flatten()
        descaled_array = self.scaler.inverse_transform(dummy_array)
        stock_price = descaled_array[:, target_col_index]

        return stock_price

    def main(self):
        for code in self.df_stock_list["code"]:
            (
                array_learn_X,
                array_learn_y,
                array_val_X,
                array_val_y,
                array_test_X,
                array_test_y,
                array_pred_X,
            ) = self.prepare_data(code)

            models, predicts, rmses = [], [], []

            for _ in range(10):
                model = self.fit(array_learn_X, array_learn_y, array_val_X, array_val_y)
                array_pred = self.predict(model, array_test_X)
                rmse = self.evaluate_model(array_pred, array_test_y)

                models.append(model)
                predicts.append(array_pred)
                rmses.append(rmse)

            champion_model = models[np.argmin(rmses)]
            future_pred = self.predict(champion_model, array_pred_X)

            actual_prices = self.inverse_transform(array_test_y)
            future_price = self.inverse_transform(future_pred)

            breakpoint()


if __name__ == "__main__":
    predictor = Predictor()
    predictor.main()
