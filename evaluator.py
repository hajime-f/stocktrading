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

    window = 30
    test_size = 30

    list_X = []
    list_y = []

    print("データを処理しています。")

    for code in stock_list["code"]:
        df = dm.load_stock_data(code, start="2019-01-01", end="2024-10-31")
        if window + 2 * test_size > len(df):
            continue

        # テクニカル指標を追加
        df = model.add_technical_indicators(df)

        # day_window日以内の終値が当日よりpercentage%以上上昇していたらフラグを立てる
        df, result = model.add_labels(df, percentage=0.5)

        for i in range(test_size, 0, -1):
            df_test = df.iloc[-window - i : -i]
            result_test = result.iloc[-window - i : -i]

            tmp_X, flag = model.prepare_input_data(df_test)
            if not flag:
                continue
            tmp_y = result_test.tail(1).values

            list_X.append(tmp_X)
            list_y.append(tmp_y)

    array_X = np.array(list_X)
    array_y = np.array(list_y)

    array_X_learn, array_X_test, array_y_learn, array_y_test = train_test_split(
        array_X, array_y, test_size=0.3, random_state=42
    )

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
    for code in stock_list["code"]:
        df = dm.load_stock_data(code, start="2024-11-01", end="end")
        if max_row < len(df):
            max_row = len(df)

    list_result = []
    list_output = []
    for i in range(max_row - window):
        print(f"{i + 1}/{max_row - window}：データを処理しています。")

        for code, brand in zip(stock_list["code"], stock_list["brand"]):
            df = dm.load_stock_data(code, start="2024-11-01", end="end")

            if len(df) < max_row:
                continue

            df = model.add_technical_indicators(df)
            df_input = df.iloc[i : i + window]
            df_output = df.iloc[i + window : i + window + 1]

            if len(df_input) < window:
                continue

            array_X, flag = model.prepare_input_data(df_input)
            if not flag:
                continue
            array_X = np.array([array_X])

            try:
                y_pred_proba = pred_model.predict(array_X, verbose=0)
            except Exception as e:
                print(e)
                continue

            try:
                list_result.append(
                    [
                        code,
                        brand,
                        y_pred_proba[0][0],
                        df_output["open"].values[0],
                        df_output["close"].values[0],
                    ]
                )
            except Exception as e:
                print(e)
                continue

        df_result = pd.DataFrame(
            list_result, columns=["code", "brand", "pred", "open", "close"]
        )

        step = 0.001
        for i in np.arange(1, 0.7, -step):
            df_extract = df_result.loc[df_result["pred"] >= i, :].copy()

            if len(df_extract) == 50:
                break

            df_extract_next = df_result.loc[df_result["pred"] >= i - step, :]
            if len(df_extract_next) > 50:
                break

        list_output.append(
            [
                df_extract["open"].sum() * 100,
                (df_extract["close"] - df_extract["open"]).sum() * 100,
            ]
        )

    df_output = pd.DataFrame(list_output, columns=["total", "result"])
    breakpoint()
