import datetime
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Bidirectional, LSTM, Dropout
from tensorflow.keras.layers import Dense, SimpleRNN, Conv1D
from tensorflow.keras.layers import GlobalMaxPooling1D, Flatten, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import gelu
from tensorflow.keras import metrics

from data_manager import DataManager
from update_model import UpdateModel

pd.set_option("display.max_rows", None)
pd.options.display.float_format = "{:.6f}".format


def compile_model(shape1, shape2):
    model = Sequential()

    model.add(InputLayer(shape=(shape1, shape2)))
    model.add(Bidirectional(SimpleRNN(200)))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", metrics.Precision(), metrics.Recall()],
    )

    return model


if __name__ == "__main__":
    dm = DataManager()
    stock_list = dm.load_stock_list()

    model = UpdateModel()
    dict_df = {}

    for code in stock_list["code"]:
        df = dm.load_stock_data(code, start="2024-05-01", end="2024-10-31")
        dict_df[f"{code}"] = model.add_technical_indicators(df)

    window = 30
    percentage = 1.0
    list_X, list_y = [], []

    for code in stock_list["code"]:
        df = dict_df[f"{code}"]
        if len(df) < window:
            continue

        for i in range(len(df) - window):
            df_input = df.iloc[i : i + window]
            df_output = df.iloc[i + window : i + window + 1]

            tmp_X, flag = model.prepare_input_data(df_input)
            if not flag:
                continue

            standard_value = df_input.tail(1)["close"].values
            flag = df_output["close"].values >= standard_value * (1 + percentage / 100)
            tmp_y = 1 if flag[0] else 0

            list_X.append(tmp_X)
            list_y.append(tmp_y)

    array_X = np.array(list_X)
    array_y = np.array(list_y)

    num = array_X.shape[0]
    num_learn = int(num * 0.7)

    array_X_learn = array_X[:num_learn]
    array_X_test = array_X[num_learn:]
    array_y_learn = array_y[:num_learn]
    array_y_test = array_y[num_learn:]

    # モデルの学習
    pred_model = compile_model(array_X.shape[1], array_X.shape[2])
    pred_model.fit(
        array_X_learn,
        array_y_learn,
        batch_size=128,
        epochs=30,
        validation_split=0.2,
        callbacks=[EarlyStopping(patience=3)],
    )

    y_pred_proba = pred_model.predict(array_X_test)

    # モデルの評価1
    y_pred = (y_pred_proba > 0.7).astype(int)
    print(classification_report(array_y_test.reshape(-1), y_pred))

    # モデルの評価2
    y_pred = (y_pred_proba > 0.8).astype(int)
    print(classification_report(array_y_test.reshape(-1), y_pred))

    # モデルの評価3
    y_pred = (y_pred_proba > 0.85).astype(int)
    print(classification_report(array_y_test.reshape(-1), y_pred))

    # そのモデルで実際にどれくらい儲かるかをバックテストする
    max_row = -1
    dict_df = {}
    for code in stock_list["code"]:
        df = dm.load_stock_data(code, start="2024-11-06", end="end")
        df = model.add_technical_indicators(df)
        if max_row < len(df):
            max_row = len(df)
        dict_df[f"{code}"] = df

    list_output = []

    for i in range(max_row - window):
        print(f"{i + 1}/{max_row - window}：データを処理しています。")

        list_result = []

        for code, brand in zip(stock_list["code"], stock_list["brand"]):
            df = dict_df[f"{code}"]

            if len(df) < max_row:
                continue

            df_input = df.iloc[i : i + window]
            df_output = df.iloc[i + window : i + window + 1]

            array_X, flag = model.prepare_input_data(df_input)
            if not flag:
                continue
            array_X = np.array([array_X])
            y_pred_proba = pred_model.predict(array_X, verbose=0)

            list_result.append(
                [
                    df_output.index[0],
                    code,
                    brand,
                    y_pred_proba[0][0],
                    df_output["open"].values[0],
                    df_output["close"].values[0],
                ]
            )

        df_result = pd.DataFrame(
            list_result, columns=["date", "code", "brand", "pred", "open", "close"]
        )

        step = 0.001
        for i in np.arange(1, 0.7, -step):
            df_extract = df_result.loc[df_result["pred"] >= i, :]

            if len(df_extract) == 50:
                break

            df_extract_next = df_result.loc[df_result["pred"] >= i - step, :]
            if len(df_extract_next) > 50:
                break

        list_output.append(
            [
                df_extract["date"],
                df_extract["open"].sum() * 100,
                (df_extract["close"] - df_extract["open"]).sum() * 100,
            ]
        )
        breakpoint()

    df_output = pd.DataFrame(list_output, columns=["date", "total", "result"])
    breakpoint()
