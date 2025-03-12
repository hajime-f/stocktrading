import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from data_management import DataManagement

pd.set_option("display.max_rows", None)


def prepare_input_data(df):
    array = np.array(df)
    scaler = StandardScaler()

    try:
        array_std = scaler.fit_transform(array)
    except ValueError:
        return None, False

    return np.array([array_std]), True


if __name__ == "__main__":
    dm = DataManagement()
    stock_list = dm.load_stock_list()

    model = load_model("./model/model_swingtrade_20250312_204936.keras")
    window = 20

    for i, code in enumerate(stock_list["code"]):
        df = dm.load_stock_data(code, start="2025-01-01", end="end")

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

        # データを絞る
        df = df.tail(window)
        if len(df) < window:
            continue

        array_X, flag = prepare_input_data(df)
        if not flag:
            continue
        y_pred = model.predict(array_X, verbose=0)
        y_pred = (y_pred > 0.6).astype(int)

        if y_pred:
            print(f"{code}, {stock_list['brand'][i]}")
