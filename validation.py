import datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    InputLayer,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from data_manager import DataManager


class Validation:
    def __init__(self):
        self.dm = DataManager()
        self.df_stock_list = self.dm.load_stock_list()

        self.train_split_ratio = 0.8
        self.threshold = 0.8
        self.window = 30

    def add_technical_indicators(self, df):
        # 日付をインデックスにする
        df.set_index("date", inplace=True)

        # 移動平均線を追加する
        df["MA5"] = df["close"].rolling(window=5).mean()
        df["MA25"] = df["close"].rolling(window=25).mean()
        df["volume_MA20"] = df["volume"].rolling(window=20).mean()

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

    def compile_model(self, shape1, shape2):
        model = Sequential()
        model.add(InputLayer(shape=(shape1, shape2)))
        model.add(Bidirectional(LSTM(200)))
        model.add(Dropout(0.3))
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(3, activation="softmax"))

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def prepare_data(self):
        scaler = StandardScaler()

        dict_df_learn = {}  # 学習用
        dict_df_test = {}  # テスト用
        dict_df_close = {}  # 答え

        today = datetime.date.today()
        start = (today - relativedelta(months=8)).strftime("%Y-%m-%d")

        for code in self.df_stock_list["code"]:
            df = self.dm.load_stock_data(code, start=start)
            df = self.add_technical_indicators(df)

            df_learn = df.iloc[:-2]
            df_test = df.iloc[:-1].tail(self.window)
            dict_df_close[code] = df["close"]

            train_split_index = int(len(df_learn) * self.train_split_ratio)
            scaler.fit(df_learn.iloc[:train_split_index])

            dict_df_learn[code] = pd.DataFrame(
                scaler.transform(df_learn), index=df_learn.index
            )
            dict_df_test[code] = pd.DataFrame(
                scaler.transform(df_test), index=df_test.index
            )

        return dict_df_learn, dict_df_test, dict_df_close

    def create_label(self, current_price, future_price, per):
        price_change_ratio = (future_price - current_price) / current_price

        if price_change_ratio >= per:
            return 2  # 上昇
        elif price_change_ratio <= -per:
            return 0  # 下落
        else:
            return 1  # 横ばい

    def fit(self, dict_df_learn, dict_df_close, per):
        list_X_train, list_y_train = [], []
        list_X_val, list_y_val = [], []
        window = self.window

        for code in dict_df_learn.keys():
            df_scaled = dict_df_learn[code]
            df_close = dict_df_close[code]

            X, y = [], []
            for i in range(len(df_scaled) - window + 1):
                window_X = df_scaled.iloc[i : i + window]

                last_date_of_window = window_X.index[-1]
                loc = df_close.index.get_loc(last_date_of_window)

                current_close = df_close.iloc[loc]
                future_close = df_close.iloc[loc + 1]
                label = self.create_label(current_close, future_close, per)

                X.append(window_X)
                y.append(label)

            split_index = int(len(X) * self.train_split_ratio)
            list_X_train.extend(X[:split_index])
            list_y_train.extend(y[:split_index])
            list_X_val.extend(X[split_index:])
            list_y_val.extend(y[split_index:])

        X_train, y_train = np.array(list_X_train), np.array(list_y_train)
        X_val, y_val = np.array(list_X_val), np.array(list_y_val)

        X_train, y_train = shuffle(X_train, y_train, random_state=42)

        model = self.compile_model(X_train.shape[1], X_train.shape[2])
        model.fit(
            X_train,
            y_train,
            batch_size=128,
            epochs=30,
            validation_data=(X_val, y_val),
            callbacks=[EarlyStopping(patience=3)],
        )

        return model

    def predict(self, model, dict_df_test, dict_df_close, per):
        list_result = []

        for code in dict_df_test.keys():
            df = dict_df_test[code]
            cl = dict_df_close[code]

            window_X = df
            array_X = np.array(window_X)

            y_pred = model.predict(np.array([array_X]), verbose=0)
            pred = np.argmax(y_pred, axis=1).item()

            last_date_of_window = window_X.index[-1]
            loc = cl.index.get_loc(last_date_of_window)

            current_close = cl.iloc[loc]
            future_close = cl.iloc[loc + 1]

            answer = self.create_label(current_close, future_close, per)
            list_result.append([pred, answer])

        return pd.DataFrame(list_result, columns=["pred", "answer"])


if __name__ == "__main__":
    val = Validation()
    per = 0.005

    # データの準備
    dict_df_learn, dict_df_test, dict_df_close = val.prepare_data()

    # モデルの学習
    model = val.fit(dict_df_learn, dict_df_close, per)

    # モデルの予測
    df_result = val.predict(model, dict_df_test, dict_df_close, per)

    pred = df_result["pred"]
    answer = df_result["answer"]

    # 結果の評価
    report = classification_report(
        answer,
        pred,
        target_names=["Negative(0)", "Neutral(1)", "Positive(2)"],
        labels=[0, 1, 2],
    )

    print(report)
