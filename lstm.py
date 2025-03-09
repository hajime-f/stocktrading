import os
from datetime import datetime
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, InputLayer
from keras.layers import Dropout

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
        return np.empty([0, 10, 15]), np.empty([0])

    if len(label_0_indices) < len(label_1_indices):
        # ラベル0のサンプル数がラベル1のサンプル数よりも少ない場合は、ダウンサンプリングを行わない
        return np.empty([0, 10, 15]), np.empty([0])

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

    array_X = np.empty([0, 10, 15])
    array_y = np.empty([0])

    for i, code in enumerate(stock_list["code"]):
        print(f"{i}/{len(stock_list)}：{code} のデータを処理しています。")

        df = dm.load_stock_data(code)

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

        # 翌営業日の終値が当日よりpercentage%以上上昇していたらフラグを立てる
        percentage = 3.0
        df_shift = df.shift(-1)
        df["increase"] = 0
        df.loc[df_shift["close"] > df["close"] * (1 + percentage / 100), "increase"] = 1

        # nan を削除
        df = df.dropna()

        # 末尾の行を削除
        try:
            df = df.drop(df.index[-1])
        except IndexError:
            continue

        window = 10
        tmp_X = prepare_input_data(df.drop("increase", axis=1), window)
        tmp_y = df["increase"].iloc[:-window].values
        tmp_X, tmp_y = downsampling(tmp_X, tmp_y)

        array_X = np.vstack((array_X, tmp_X))
        array_y = np.hstack((array_y, tmp_y))

    array_X_learn, array_X_test, array_y_learn, array_y_test = train_test_split(
        array_X, array_y, test_size=0.3, random_state=42
    )

    # モデルの学習
    pred_model = PredictionModel()
    model = pred_model.DNN_compile(array_X_learn)
    model.fit(array_X_learn, array_y_learn, batch_size=64, epochs=10)

    # モデルの評価
    y_pred = model.predict(array_X_test)
    y_pred = (y_pred > 0.5).astype(int)

    print("accuracy = ", accuracy_score(y_true=array_y_test, y_pred=y_pred))
    print(classification_report(array_y_test, y_pred))

    # モデルの保存
    now = datetime.now()
    filename = now.strftime("model_swingtrade_%Y%m%d_%H%M%S.keras")

    dirname = "./model"
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    filename = os.path.join(dirname, filename)
    model.save(filename)

    # モデルの読み込み
    model = load_model(filename)

    # モデルの再評価
    y_pred = model.predict(array_X_test)
    y_pred = (y_pred > 0.5).astype(int)

    print("accuracy = ", accuracy_score(y_true=array_y_test, y_pred=y_pred))
    print(classification_report(array_y_test, y_pred))
