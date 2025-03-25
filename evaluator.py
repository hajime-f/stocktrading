import numpy as np

from keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import gelu
from tensorflow.keras import metrics

from data_manager import DataManager
from update_model import UpdateModel


def compile_model(self, shape1, shape2):
    model = Sequential()

    model.add(InputLayer(shape=(shape1, shape2)))
    model.add(
        Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.001)))
    )
    model.add(Dropout(0.4))
    model.add(Bidirectional(LSTM(128, kernel_regularizer=l2(0.001))))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation=gelu))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(learning_rate=0.0005)

    model.compile(
        optimizer=optimizer,
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

    for code, brand in zip(stock_list["code"], stock_list["brand"]):
        df = dm.load_stock_data(code, start="2019-01-01", end="end")

        # テクニカル指標を追加
        df = model.add_technical_indicators(df)

        # day_window日以内の終値が当日よりpercentage%以上上昇していたらフラグを立てる
        df, result = model.add_labels(df)

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

    pred_model = model.compile_model(array_X.shape[1], array_X.shape[2])
    pred_model.fit(
        array_X,
        array_y,
        batch_size=128,
        epochs=30,
        validation_split=0.2,
        callbacks=[EarlyStopping(patience=3)],
    )
