from sklearn.svm import SVC

from data_manager import DataManager


class Classifier:
    def __init__(self):
        self.dm = DataManager()
        self.df = self.dm.load_aggregate()

    def prepare_data(self):
        df_tmp = self.df[["threshold", "long", "short"]]

        df_tmp.loc[:, "threshold"] = (
            df_tmp["threshold"] - df_tmp["threshold"].mean()
        ) / df_tmp["threshold"].std()
        df_tmp.loc[:, "long"] = (df_tmp["long"] - df_tmp["long"].mean()) / df_tmp[
            "long"
        ].std()
        df_tmp.loc[:, "short"] = (df_tmp["short"] - df_tmp["short"].mean()) / df_tmp[
            "short"
        ].std()

        df_input = df_tmp[:-1]
        df_test = df_tmp.tail(1)

        self.df["result"] = (self.df["pl_long"] > self.df["pl_short"]).astype(int)
        df_output = self.df[:-1]["result"]

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
    pred = clf.predict(df_test)
    print(pred)
