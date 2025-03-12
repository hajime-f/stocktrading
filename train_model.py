import os
import pickle
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from keras.models import load_model

from model import ModelLibrary


if __name__ == "__main__":
    # データファイル名
    filename_list = [
        "data_20250310_153016.pkl",
        "data_20250311_153133.pkl",
        "data_20250312_153037.pkl",
    ]
    df_list = []
    n_symbols = 0

    for i, filename in enumerate(filename_list):
        print(f"{filename} からデータを読み込んでいます。")
        filepath_name = os.path.join("./data/", filename)
        with open(filepath_name, "rb") as f:
            tmp = pickle.load(f)
            df_list += tmp
        n_symbols += len(tmp)

    # モデルライブラリを初期化する
    model = ModelLibrary()

    print("データに特徴を追加しています。")
    df_list = model.add_technical_indicators(df_list)

    print("データにラベルを追加しています。")
    label_list = model.add_labels(df_list)

    print("学習用データと検証用データを準備しています。")
    X, y = model.prepare_data(df_list, label_list)
    X_learn, X_test, y_learn, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print("予測モデルをコンパイルしています。")
    pred_model = model.compile_model(X.shape[1], X.shape[2])

    print("予測モデルを学習させています。")
    pred_model.fit(X_learn, y_learn, epochs=10, batch_size=64)

    print("予測モデルを評価しています。")
    pred = pred_model.predict(X_test)
    pred = (pred > 0.5).astype(int)

    print("accuracy = ", accuracy_score(y_true=y_test, y_pred=pred))
    print(classification_report(y_test, pred))

    print("学習済みモデルを保存しています。")
    filename = model.save_model(pred_model)

    print("学習済みモデルを復元しています。")
    pred_model = model.load_model(filename)

    print("予測モデルを再評価しています。")
    pred = pred_model.predict(X_test)
    pred = (pred > 0.5).astype(int)

    print("accuracy = ", accuracy_score(y_true=y_test, y_pred=pred))
    print(classification_report(y_test, pred))
