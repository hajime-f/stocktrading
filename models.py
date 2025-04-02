import sqlite3

from data_manager import DataManager
from model_manager import ModelManager


class LongModel(ModelManager):
    def __init__(self):
        super().__init__()
        self.dm = DataManager()

        percentage = 0.5
        self.up_per = 1 + percentage / 100

    def fit(self):
        model_names = self.fit(self.up_per)

        conn = sqlite3.connect(f"{self.dm.base_dir}/data/stock_data.db")
        with conn:
            model_names.to_sql("Models_2", conn, if_exists="replace", index=False)

    def predict(self):
        df = self.predict("Models_2")

        conn = sqlite3.connect(f"{self.dm.base_dir}/data/stock_data.db")
        with conn:
            df.to_sql("Target_2", conn, if_exists="append", index=False)

        return df


class ShortModel(ModelManager):
    def __init__(self):
        super().__init__()
        self.dm = DataManager()

        percentage = 0.5
        self.down_per = 1 - percentage / 100

    def fit(self):
        model_names = self.fit(self.down_per)

        conn = sqlite3.connect(f"{self.dm.base_dir}/data/stock_data.db")
        with conn:
            model_names.to_sql("Models_3", conn, if_exists="replace", index=False)

    def predict(self):
        df = self.predict("Models_3")

        conn = sqlite3.connect(f"{self.dm.base_dir}/data/stock_data.db")
        with conn:
            df.to_sql("Target_3", conn, if_exists="append", index=False)

        return df
