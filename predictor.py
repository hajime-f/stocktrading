from datetime import datetime
import sqlite3
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from keras.models import load_model

from data_manager import DataManager


class Predictor:
    def __init__(self, filename):
        self.model = load_model(filename)
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

        return np.array([array_std]), True

    def predict(self, data):
        return self.model.predict(data, verbose=0)


if __name__ == "__main__":
    dm = DataManager()
    list_stocks = dm.load_stock_list()
    df_models = dm.load_model_list()

    list_result = []

    for filename, window in zip(df_models["model_name"], df_models["window"]):
        filename = os.path.join(f"{dm.base_dir}/model/", filename)

        predictor = Predictor(filename)

        # データを読み込む
        for code, brand in zip(list_stocks["code"], list_stocks["brand"]):
            df = dm.load_stock_data(code, start="2024-12-01", end="end")

            # テクニカル指標を追加
            df = predictor.add_technical_indicators(df)

            # データを絞る
            df = df.tail(window)
            if len(df) < window:
                continue

            # 予測する
            array_X, flag = predictor.prepare_input_data(df)
            if not flag:
                continue
            y_pred = predictor.predict(array_X)

            list_result.append([code, brand, y_pred[0][0]])

    df_result = pd.DataFrame(list_result, columns=["code", "brand", "pred"])
    df_result = df_result.loc[df_result.groupby("code")["pred"].idxmax(), :]

    step = 0.001
    for i in np.arange(1, 0.7, -step):
        df_extract = df_result.loc[df_result["pred"] >= i, :].copy()

        if len(df_extract) == 50:
            break

        df_extract_next = df_result.loc[df_result["pred"] >= i - step, :]
        if len(df_extract_next) > 50:
            break

    df_extract.loc[:, "date"] = datetime.now().strftime("%Y-%m-%d")
    df_extract = df_extract[["date", "code", "brand", "pred"]]

    conn = sqlite3.connect(f"{dm.base_dir}/data/stock_data.db")
    with conn:
        df_extract.to_sql("Target", conn, if_exists="append", index=False)
