import pandas as pd
import numpy as np

from rich.console import Console
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

console = Console(log_time_format="%Y-%m-%d %H:%M:%S")
pd.set_option("display.max_rows", None)
pd.options.display.float_format = "{:.6f}".format


def compile_model(shape1, shape2, model_num):
    console.log(f"学習モデル：{model_num}")

    model = Sequential()
    model.add(InputLayer(shape=(shape1, shape2)))

    if model_num == 1:
        model.add(Bidirectional(SimpleRNN(200)))
        model.add(Dropout(0.3))
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation="sigmoid"))

    elif model_num == 2:
        model.add(Bidirectional(LSTM(200)))
        model.add(Dropout(0.3))
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation="sigmoid"))

    elif model_num == 3:
        model.add(Conv1D(200, 3, activation=gelu, kernel_regularizer=l2(0.01)))
        model.add(MaxPooling1D(3))
        model.add(Conv1D(200, 3, activation=gelu, kernel_regularizer=l2(0.01)))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation="sigmoid"))

    elif model_num == 4:
        model.add(Conv1D(filters=64, kernel_size=3, activation="relu"))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.3))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(100, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation="sigmoid"))

    elif model_num == 5:
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Dropout(0.4))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation="sigmoid"))

    else:
        raise ValueError("model_num が不正です。")

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", metrics.Precision(), metrics.Recall()],
    )

    return model


def process(model_num):
    dm = DataManager()
    stock_list = dm.load_stock_list()

    console.log("データを読み込んでいます。")
    model = UpdateModel()
    dict_df = {}

    for code in stock_list["code"]:
        df = dm.load_stock_data(code, start="2024-05-01", end="2024-10-31")
        dict_df[f"{code}"] = model.add_technical_indicators(df)

    console.log("データを処理しています。")
    window = 30
    percentage = 0.5
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
    console.log("モデルを学習させています。")
    pred_model = compile_model(array_X.shape[1], array_X.shape[2], model_num)
    pred_model.fit(
        array_X_learn,
        array_y_learn,
        batch_size=128,
        epochs=30,
        validation_split=0.2,
        callbacks=[EarlyStopping(patience=3)],
        verbose=0,
    )

    y_pred_proba = pred_model.predict(array_X_test, verbose=0)

    # モデルの評価1
    y_pred = (y_pred_proba > 0.7).astype(int)
    print(classification_report(array_y_test, y_pred))

    # モデルの評価2
    y_pred = (y_pred_proba > 0.8).astype(int)
    print(classification_report(array_y_test, y_pred))

    # モデルの評価3
    y_pred = (y_pred_proba > 0.85).astype(int)
    print(classification_report(array_y_test, y_pred))

    # そのモデルで実際にどれくらい儲かるかをバックテストする
    console.log("データを読み込んでいます。")
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
        console.log(f"{i + 1}/{max_row - window}：データを処理しています。")

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
        for i in np.arange(1, 0.70, -step):
            df_extract = df_result.loc[df_result["pred"] >= i, :].copy()

            if len(df_extract) == 50:
                break

            df_extract_next = df_result.loc[df_result["pred"] >= i - step, :]
            if len(df_extract_next) > 50:
                break
        try:
            list_output.append(
                [
                    df_extract["date"].iloc[0],
                    df_extract["open"].sum() * 100,
                    (df_extract["close"] - df_extract["open"]).sum() * 100,
                ]
            )
        except Exception:
            pass

    df_output = pd.DataFrame(list_output, columns=["date", "total", "result"])
    df_output.set_index("date", inplace=True)
    pd.options.display.float_format = "{:.0f}".format

    console.log(df_output)
    console.log(df_output["result"].mean())


if __name__ == "__main__":
    model_num = [1, 2, 3, 4, 5]

    for n in model_num:
        process(n)
