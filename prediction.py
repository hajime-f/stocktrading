import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from data_management import DataManagement

pd.set_option("display.max_rows", None)


def prepare_input_data(df, window=10):
    X_list = []

    df = np.array(df)
    scaler = StandardScaler()

    try:
        df_std = scaler.fit_transform(df)
    except ValueError:
        return None, False

    X_list.append(df_std)
    return np.array(X_list), True


if __name__ == "__main__":
    dm = DataManagement()
    stock_list = dm.load_stock_list()

    model = load_model("./model/model_swingtrade_20250310_182001.keras")

    for i, code in enumerate(stock_list["code"]):
        df = dm.load_stock_data(code).tail(10 * 5)

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

        # nan を削除
        df = df.dropna()

        # データを絞る
        df = df.tail(10)

        array_X, flag = prepare_input_data(df, 10)
        if not flag:
            continue
        y_pred = model.predict(array_X, verbose=0)
        y_pred = (y_pred > 0.999).astype(int)

        if y_pred:
            print(f"{code}, {stock_list['brand'][i]}")
