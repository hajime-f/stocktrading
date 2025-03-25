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

    for i, code in enumerate(stock_list["code"]):
        print(f"{i + 1}/{len(stock_list)}：{code} のデータを処理しています。")

        df = dm.load_stock_data(code, start="2019-01-01", end="end")
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
    y_pred = (y_pred_proba > 0.5).astype(int)
    print(classification_report(array_y_test.reshape(-1), y_pred))

    # モデルの評価2
    y_pred = (y_pred_proba > 0.7).astype(int)
    print(classification_report(array_y_test.reshape(-1), y_pred))

    # モデルの評価3
    y_pred = (y_pred_proba > 0.8).astype(int)
    print(classification_report(array_y_test.reshape(-1), y_pred))

    # モデルの評価4
    y_pred = (y_pred_proba > 0.85).astype(int)
    print(classification_report(array_y_test.reshape(-1), y_pred))

    # モデルの評価5
    y_pred = (y_pred_proba > 0.9).astype(int)
    print(classification_report(array_y_test.reshape(-1), y_pred))
