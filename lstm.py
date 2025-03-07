import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer
from keras.layers import Dropout
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt

from data_management import DataManagement

pd.set_option("display.max_rows", None)


def prepare_input_data(df, window=10):
    X_list = []

    df = np.array(df)
    scaler = StandardScaler()

    for i in range(len(df) - window):
        df_s = df[i : i + window]
        df_std = scaler.fit_transform(df_s)
        X_list.append(df_std)

    return np.array(X_list)


class PredictionModel:
    def __init__(self):
        pass

    def DNN_compile(self, array):
        model = Sequential()

        model.add(InputLayer(shape=(array.shape[1], array.shape[2])))
        model.add(LSTM(256, activation="relu"))
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
    df = dm.load_stock_data(1301)

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
    df.loc[df_shift["close"] > df["close"], "increase"] = 1

    # nan を削除
    df = df.dropna()

    # 学習用データとテスト用データに分割
    df_learn = df[(df.index >= "2021-03-01") & (df.index <= "2024-06-30")]
    df_test = df[(df.index >= "2024-07-01") & (df.index <= "2025-03-06")]

    # 入出力データを準備
    window = 10
    array_learn_X = prepare_input_data(df_learn, window)
    array_learn_y = df_learn["increase"].iloc[:-window].values
    array_test_X = prepare_input_data(df_test, window)
    array_test_y = df_test["increase"].iloc[:-window].values

    test_scores = []
    tss = TimeSeriesSplit(n_splits=4)
    pred_model = PredictionModel()
    scaler = StandardScaler()

    for fold, (learn_indices, test_indices) in enumerate(tss.split(array_learn_X)):
        array_learn_X_validate, array_test_X_validate = (
            array_learn_X[learn_indices],
            array_learn_X[test_indices],
        )
        array_learn_y_validate, array_test_y_validate = (
            array_learn_y[learn_indices],
            array_learn_y[test_indices],
        )

        model = pred_model.DNN_compile(array_learn_X_validate)

        model.fit(
            array_learn_X_validate, array_learn_y_validate, epochs=10, batch_size=64
        )
        y_test_pred = model.predict(array_test_X_validate)
        y_test_pred = np.where(y_test_pred < 0.5, 0, 1)
        score = accuracy_score(array_test_y_validate, y_test_pred)
        print(f"fold {fold} MAE: {score}")

        test_scores.append(score)

    print(f"test_scores: {test_scores}")
    print(f"CV score: {np.mean(test_scores)}")

    # モデルの学習
    model = pred_model.DNN_compile(array_learn_X)
    model.fit(array_learn_X, array_learn_y, batch_size=64, epochs=10)

    # モデルの評価
    y_pred = model.predict(array_test_X)
    y_pred = (y_pred > 0.5).astype(int)

    print("accuracy = ", accuracy_score(y_true=array_test_y, y_pred=y_pred))
    print(classification_report(array_test_y, y_pred))

    breakpoint()
