import os
from datetime import datetime
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Dropout

import matplotlib.pyplot as plt
import mplfinance as mpf

from data_management import DataManagement

pd.set_option("display.max_rows", None)

window = 20
test_size = 30


def prepare_input_data(df):
    array = np.array(df)
    scaler = StandardScaler()

    try:
        array_std = scaler.fit_transform(array)
    except ValueError:
        return None, False

    return np.array([array_std]), True


def downsampling(X_array, y_array):
    """
    ラベルの偏りが大きいので、ダウンサンプリングを行う
    （0が多く1が少ないことを前提としているので、逆になるとエラーが出ることに注意）
    """
    # ラベル1のインデックスを取得
    label_1_indices = np.where(y_array == 1)[0]
    # ラベル0のインデックスを取得
    label_0_indices = np.where(y_array == 0)[0]

    if len(label_1_indices) == 0:
        # ラベル1のサンプルが0個の場合は、ダウンサンプリングを行わない
        return np.empty([0, window, 15]), np.empty([0])

    if len(label_0_indices) < len(label_1_indices):
        # ラベル0のサンプル数がラベル1のサンプル数よりも少ない場合は、ダウンサンプリングを行わない
        return np.empty([0, window, 15]), np.empty([0])

    # ラベル1のサンプル数と同じ数だけ、ラベル0のデータをランダムにサンプリング
    downsampled_label_0_indices = resample(
        label_0_indices,
        replace=False,
        n_samples=len(label_1_indices),
        random_state=42,
    )

    # ダウンサンプリングしたデータのインデックスとラベル1のインデックスを結合
    selected_indices = np.concatenate([downsampled_label_0_indices, label_1_indices])

    # X_arrayとy_arrayから選択したインデックスのデータを取得
    X_downsampled = X_array[selected_indices]
    y_downsampled = y_array[selected_indices]

    return X_downsampled, y_downsampled


def add_technical_indicators(df):
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


def add_labels(df, percentage=1.0, day_window=3):
    # shifted_df = df.shift(-day_window)
    # df["increase"] = 0
    # df.loc[shifted_df["close"] > df["close"] * (1 + percentage / 100), "increase"] = 1
    # df = df.iloc[:-day_window]

    # return df

    result = pd.DataFrame(np.zeros((len(df), 1)), columns=["increase"])

    for i in range(day_window):
        shifted_df = df.shift(-(i + 1))
        result[f"increase_{i + 1}"] = 0
        result.loc[
            shifted_df["close"] > df["close"] * (1 + percentage / 100),
            f"increase_{i + 1}",
        ] = 1

    result["increase"] = 0
    for i in range(day_window):
        result["increase"] += result[f"increase_{i + 1}"]
        result.drop(f"increase_{i + 1}", axis=1, inplace=True)

    result.loc[result["increase"] > 0, "increase"] = 1
    result.index = df.index

    return pd.concat([df, result], axis=1).iloc[:-day_window]


def candle_plot(df):
    df_candle = df[["open", "high", "low", "close", "volume"]]
    df_candle.columns = ["Open", "High", "Low", "Close", "Volume"]
    df_candle.index = pd.to_datetime(df_candle.index)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df["MA5"], color="r", label="MA5")
    ax.plot(df["MA25"], color="g", label="MA25")
    ax.fill_between(df.index, df["Upper"], df["Lower"], color="gray", alpha=0.5)
    mpf.plot(df_candle, type="candle", volume=False, style="default", ax=ax)
    plt.axis("off")
    plt.show()


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
    stock_list = dm.load_stock_list()

    array_X = np.empty([0, window, 21])
    array_y = np.empty([0])

    for i, code in enumerate(stock_list["code"]):
        print(f"{i + 1}/{len(stock_list)}：{code} のデータを処理しています。")

        # データを読み込む
        df = dm.load_stock_data(code, start="2020-01-01", end="2023-12-31")
        if window * test_size > len(df):
            continue

        # テクニカル指標を追加
        df = add_technical_indicators(df)
        df = df.reset_index(drop=True)

        # day_window日以内の終値が当日よりpercentage%以上上昇していたらフラグを立てる
        percentage, day_window = 0.5, 1
        df = add_labels(df, percentage=percentage, day_window=day_window)

        for j in range(test_size, 0, -1):
            df_test = df.iloc[-window - j : -j].drop("date", axis=1)

            tmp_X, flag = prepare_input_data(df_test.drop("increase", axis=1))
            if not flag:
                continue
            tmp_y = df_test.tail(1)["increase"].values

            # if tmp_y:
            #     df_plot = df.iloc[-window - j : -j].set_index("date")
            #     candle_plot(df_plot)

            array_X = np.vstack((array_X, tmp_X))
            array_y = np.hstack((array_y, tmp_y))

    # array_X, array_y = downsampling(array_X, array_y)

    array_X_learn, array_X_test, array_y_learn, array_y_test = train_test_split(
        array_X, array_y, test_size=0.3, random_state=42
    )

    # モデルの学習
    pred_model = PredictionModel()
    model = pred_model.DNN_compile(array_X_learn)
    model.fit(
        array_X_learn,
        array_y_learn,
        batch_size=128,
        epochs=10,
    )

    # モデルの評価１
    y_pred = model.predict(array_X_test)
    y_pred = (y_pred > 0.7).astype(int)

    print(classification_report(array_y_test, y_pred))

    # モデルの評価２
    y_pred = model.predict(array_X_test)
    y_pred = (y_pred > 0.75).astype(int)

    print(classification_report(array_y_test, y_pred))

    # モデルの評価３
    y_pred = model.predict(array_X_test)
    y_pred = (y_pred > 0.8).astype(int)

    print(classification_report(array_y_test, y_pred))

    # モデルの評価４
    y_pred = model.predict(array_X_test)
    y_pred = (y_pred > 0.85).astype(int)

    print(classification_report(array_y_test, y_pred))

    # モデルの保存
    now = datetime.now()
    filename = now.strftime("model_swingtrade_%Y%m%d_%H%M%S.keras")

    dirname = "./model"
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    filename = os.path.join(dirname, filename)
    model.save(filename)
