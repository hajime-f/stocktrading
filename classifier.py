from sklearn.svm import SVC

from data_manager import DataManager


class Classifier:
    def __init__(self):
        self.dm = DataManager()
        self.df = self.dm.load_aggregate()

    def prepare_data(self):
        self.df["result"] = (self.df["pl_long"] > self.df["pl_short"]).astype(int)

        # 変数のスケーリング
        df_tmp = self.df[["threshold", "long", "short"]]
        df_tmp = (df_tmp - df_tmp.mean()) / df_tmp.std()

        df_input = df_tmp[:-1]
        df_output = self.df[:-1]["result"]
        df_test = df_tmp.tail(1)

        return df_input, df_output, df_test

    def fit(self, df_input, df_output):
        self.clf = SVC(kernel="linear")
        self.clf.fit(df_input, df_output)

    def predict(self, df_test):
        return self.clf.predict(df_test)


if __name__ == "__main__":
    clf = Classifier()

    df_input, df_output, df_test = clf.prepare_data()

    clf.fit(df_input, df_output)
    y_pred = clf.predict(df_test)
    print(y_pred)
