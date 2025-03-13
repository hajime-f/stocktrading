import os
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, InputLayer, Dropout

pd.set_option("display.max_rows", None)


class ModelLibrary:
    def __init__(self):
        self.clf = None

    def add_technical_indicators(self, df_list):
        # 移動平均を計算する
        df_list = [self.calc_moving_average(df) for df in df_list]

        # MACDを計算する
        df_list = [self.calc_macd(df) for df in df_list]

        # ボリンジャーバンドを計算する
        df_list = [self.calc_bollinger_band(df) for df in df_list]

        # RSIを計算する
        df_list = [self.calc_rsi(df) for df in df_list]

        return df_list

    def calc_moving_average(self, data):
        data["MA5"] = data["close"].rolling(window=5).mean()
        data["MA25"] = data["close"].rolling(window=25).mean()

        return data

    def calc_macd(self, data):
        data["MACD"] = (
            data["close"].ewm(span=12).mean() - data["close"].ewm(span=26).mean()
        )
        data["SIGNAL"] = data["MACD"].ewm(span=9).mean()
        data["HISTOGRAM"] = data["MACD"] - data["SIGNAL"]

        return data

    def calc_bollinger_band(self, data):
        sma20 = data["close"].rolling(window=20).mean()
        std20 = data["close"].rolling(window=20).std()
        data["Upper"] = sma20 + (std20 * 2)
        data["Lower"] = sma20 - (std20 * 2)

        return data

    def calc_rsi(self, data):
        delta = data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data["RSI"] = (100 - (100 / (1 + rs))) / 100

        return data

    def add_labels(self, df_list):
        # 正解ラベルを作成する
        label_list = [self.check_price_change(df, 0.5, 20) for df in df_list]

        return label_list

    def check_price_change(self, df, percentage=0.5, time_window=20):
        # ある時刻における株価を基準にして、そこからtime_window分以内にpercentage％上昇するか否かを判定する。
        # target_price = base_price + abs(base_price) * (percentage / 100)

        result = pd.DataFrame(np.zeros((len(df), 1)), columns=["Result"])

        for i in range(time_window):
            shifted_df = df.shift(-(i + 1))
            result[f"Result_{i + 1}"] = 0
            condition = (
                shifted_df["close"] > df["close"] * (1 + percentage / 100)
            ).values
            result.loc[condition, f"Result_{i + 1}"] = 1

        for i in range(time_window):
            result["Result"] += result[f"Result_{i + 1}"]
            result.drop(f"Result_{i + 1}", axis=1, inplace=True)

        result.loc[result["Result"] > 0, "Result"] = 1

        return result

    def prepare_data(self, df_list, label_list, window=10):
        # 学習用データと検証用データを準備する

        df_list = [df.dropna() for df in df_list]
        label_list = [label.dropna() for label in label_list]

        scaler = StandardScaler()

        list_X = []
        list_y = []

        for df, label in zip(df_list, label_list):
            array_X = np.array(df)
            array_y = np.array(label)

            for i in range(len(df) - window):
                tmp1 = scaler.fit_transform(array_X[i : i + window])
                tmp2 = array_y[i + window - 1]

                list_X.append(tmp1)
                list_y.append(tmp2)

        array_X = np.array(list_X)
        array_y = np.array(list_y)

        # array_X_downsampled, array_y_downsampled = self.downsampling(array_X, array_y)

        # return array_X_downsampled, array_y_downsampled

        return array_X, array_y

    def downsampling(self, X_array, y_array):
        """
        ラベルの偏りが大きいので、ダウンサンプリングを行う
        （0が多く1が少ないことを前提としているので、逆になるとエラーが出ることに注意）
        """
        # ラベル1のインデックスを取得
        label_1_indices = np.where(y_array == 1)[0]
        # ラベル0のインデックスを取得
        label_0_indices = np.where(y_array == 0)[0]

        if len(label_1_indices) == 0:
            # ラベル1のサンプルが0個の場合は、例外を発生させる
            raise ValueError("ラベル1のサンプルが0個です。")

        # ラベル1のサンプル数と同じ数だけ、ラベル0のデータをランダムにサンプリング
        downsampled_label_0_indices = resample(
            label_0_indices,
            replace=False,
            n_samples=len(label_1_indices),
            random_state=42,
        )

        # ダウンサンプリングしたデータのインデックスとラベル1のインデックスを結合
        selected_indices = np.concatenate(
            [downsampled_label_0_indices, label_1_indices]
        )

        # X_arrayとy_arrayから選択したインデックスのデータを取得
        X_downsampled = X_array[selected_indices]
        y_downsampled = y_array[selected_indices]

        return X_downsampled, y_downsampled

    def compile_model(self, shape1, shape2):
        model = Sequential()

        model.add(InputLayer(shape=(shape1, shape2)))
        model.add(LSTM(256, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        return model

    def save_model(self, model):
        now = datetime.now()
        filename = now.strftime("model_daytrade_%Y%m%d_%H%M%S.keras")

        dirname = "./model"
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        filename = os.path.join(dirname, filename)

        model.save(filename)

        return filename

    def load_model(self, filename):
        self.model = load_model(filename)
        return self.model

    def predict(self, data):
        return self.model.predict(data, verbose=0)
