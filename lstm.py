import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dropout
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt

from data_management import DataManagement

pd.set_option("display.max_rows", None)


class PredictionModel:
    def __init__(self):
        pass

    def DNN_compile(self, X_df):
        model = Sequential()

        model.add(LSTM(256, activation="relu", input_shape=(X_df.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        return model


if __name__ == "__main__":
    dm = DataManagement()
    df = dm.load_stock_data(2158)

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

    # 翌営業日の終値が当日より2.5%以上上昇していたらフラグを立てる
    df_shift = df.shift(-1)
    df["increase"] = 0
    df.loc[df_shift["close"] > df["close"] * 1.025, "increase"] = 1

    # 学習用データを準備する
    df_learn = df[(df.index >= "2021-03-01") & (df.index <= "2024-06-30")]
    X_learn = df_learn.drop(columns=["increase"]).iloc[:-1]
    y_learn = df_learn["increase"].iloc[:-1]

    # # テスト用データを準備する
    # df_test = df[(df["date"] >= "2024-07-01") & (df["date"] <= "2025-03-05")]
    # X_test = df_test.drop(columns=["date"]).iloc[:-1]
    # y_test = df_test["close"].shift(-1).iloc[:-1]

    # # 標準化
    # scaler = StandardScaler()
    # X_learn = scaler.fit_transform(X_learn)
    # X_test = scaler.transform(X_test)

    # pred_model = PredictionModel()

    # # モデルの学習
    # model = pred_model.DNN_compile(X_learn)
    # model.fit(X_learn, y_learn, batch_size=64, epochs=10)

    # # モデルの評価
    # Y_pred = model.predict(X_test)
    # Y_pred = (Y_pred > 0.5).astype(int)

    # print("accuracy = ", accuracy_score(y_true=y_test, y_pred=Y_pred))
