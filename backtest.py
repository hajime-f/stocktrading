import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from keras.models import load_model

from data_management import DataManagement

pd.set_option("display.max_rows", None)


class Backtest:
    def __init__(self, window=10, test_size=10):
        self.dm = DataManagement()
        self.stock_list = self.dm.load_stock_list()

        filename = os.path.join("./model/", "model_swingtrade_20250312_002952.keras")
        self.model = load_model(filename)

        self.window = window
        self.test_size = test_size

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

    def add_labels(self, df, percentage=1.0, day_window=3):
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

        return pd.concat([df, result], axis=1).iloc[:-day_window]

    def prepare_input_data(self, df):
        array = np.array(df)

        try:
            array_std = self.scaler.fit_transform(array)
        except ValueError:
            return None, False

        return np.array([array_std]), True


if __name__ == "__main__":
    window = 20
    test_size = 10

    bt = Backtest(window, test_size)

    array_y_stack = np.empty([0])
    pred_stack = np.empty([0])

    for i, code in enumerate(bt.stock_list["code"]):
        print(f"{i + 1}/{len(bt.stock_list)}：{code} のデータを処理しています。")

        # データを読み込む
        df = bt.dm.load_stock_data(code, start="2024-01-01", end="end")

        # テクニカル指標を追加する
        df = bt.add_technical_indicators(df)

        # day_window日以内の終値が当日よりpercentage%以上上昇していたらフラグを立てる
        percentage, day_window = 1.0, 3
        df = bt.add_labels(df, percentage=percentage, day_window=day_window)

        for j in range(test_size, 0, -1):
            df_test = df.iloc[-window - j : -j]

            array_X, flag = bt.prepare_input_data(df_test.drop("increase", axis=1))
            if not flag:
                continue
            array_y = df_test.tail(1)["increase"].values

            pred = bt.model.predict(array_X, verbose=0)
            pred = (pred > 0.5).astype(int)

            array_y_stack = np.append(array_y_stack, array_y)
            pred_stack = np.append(pred_stack, pred)

    print(classification_report(array_y_stack, pred_stack))

    # for i, code in enumerate(bt.stock_list["code"]):
    #     # データを読み込む
    #     df_test = bt.dm.load_stock_data(code).tail(window * 5)

    #     # テクニカル指標を追加する
    #     df_test = bt.add_technical_indicators(df_test)

    #     # データを絞る
    #     df_test = df_test.tail(window)

    #     array_X, flag = bt.prepare_input_data(df_test, window)
    #     if not flag:
    #         continue

    #     pred = bt.model.predict(array_X, verbose=0)
    #     pred = (pred > 0.5).astype(int)

    #     if pred[0][0]:
    #         print(f"{code}, {bt.stock_list['brand'][i]}")
