import pandas as pd
from sklearn.svm import SVC

from data_manager import DataManager


class Classifier:
    def __init__(self):
        pass


if __name__ == "__main__":
    dm = DataManager()

    df = dm.load_aggregate()

    df["pl_long"] = pd.to_numeric(df["pl_long"], errors="coerce")
    df["pl_short"] = pd.to_numeric(df["pl_short"], errors="coerce")
    df["result"] = (df["pl_long"] > df["pl_short"]).astype(int)

    df_input = df[:-1][["threshold", "long", "short"]]
    df_output = df[:-1]["result"]
    df_test = df.tail(1)[["threshold", "long", "short"]]

    clf = SVC(kernel="linear")
    clf.fit(df_input, df_output)

    y_pred = clf.predict(df_test)

    breakpoint()
