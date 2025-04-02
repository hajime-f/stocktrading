import sqlite3

from data_manager import DataManager
from model_manager import ModelManager


class LongModel(ModelManager):
    def __init__(self):
        super().__init__()
        self.dm = DataManager()

    def fit(self, percentage):
        up_per = 1 + percentage / 100
        df_models = super().fit(up_per)

        return df_models

    def predict(self, df_models):
        df = super().predict(df_models)
        df.loc[:, "side"] = 2

        conn = sqlite3.connect(f"{self.dm.base_dir}/data/stock_data.db")
        with conn:
            df.to_sql("Target_Long", conn, if_exists="append", index=False)

        return df


class ShortModel(ModelManager):
    def __init__(self):
        super().__init__()
        self.dm = DataManager()

    def fit(self, percentage):
        down_per = 1 - percentage / 100
        df_models = super().fit(down_per)

        return df_models

    def predict(self, df_models):
        df = super().predict(df_models)
        df.loc[:, "side"] = 1

        conn = sqlite3.connect(f"{self.dm.base_dir}/data/stock_data.db")
        with conn:
            df.to_sql("Target_Short", conn, if_exists="append", index=False)

        return df
