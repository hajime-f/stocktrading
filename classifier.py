from sklearn.svm import SVC

from data_manager import DataManager


class Classifier:
    def __init__(self):
        self.dm = DataManager()
        self.df = self.dm.load_aggregate()

    def fit(self, df_input, df_output):
        self.clf = SVC(kernel="linear")
        self.clf.fit(df_input, df_output)

    def predict(self, df_test):
        y_pred = self.clf.predict(df_test)
        return y_pred


if __name__ == "__main__":
    clf = Classifier()

    clf.df["result"] = (clf.df["pl_long"] > clf.df["pl_short"]).astype(int)
    df_input = clf.df[:-1][["threshold", "long", "short"]]
    df_output = clf.df[:-1]["result"]
    df_test = clf.df.tail(1)[["threshold", "long", "short"]]

    clf.fit(df_input, df_output)
    y_pred = clf.predict(df_test)
    print(y_pred)
