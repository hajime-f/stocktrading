import datetime
import sqlite3

from data_manager import DataManager
from misc import Misc


class Aggregator:
    def __init__(self):
        self.dm = DataManager()
        self.today = datetime.date.today().strftime("%Y-%m-%d")

    def aggregate(self, table_name):
        df = self.dm.load_table_by_date(table_name, self.today)
        pl = 0

        for index, row in df.iterrows():
            oc_prices = self.dm.load_open_close_prices(row["code"], self.today)
            if not oc_prices.empty:
                if table_name == "Target_Long":
                    pl += oc_prices["close"] - oc_prices["open"]
                elif table_name == "Target_Short":
                    pl += oc_prices["open"] - oc_prices["close"]
                else:
                    raise ValueError("不正なテーブル名が指定されました。")

        return pl.item() * 100

    def save_pl(self, pl_long, pl_short):
        conn = sqlite3.connect(self.dm.db)
        with conn:
            sql = """
            UPDATE Aggregate SET pl_long = ?, pl_short = ?
            WHERE date = ?;
            """
            conn.execute(sql, (pl_long, pl_short, self.today))


if __name__ == "__main__":
    if Misc().check_day_type(datetime.date.today()):
        exit()

    agg = Aggregator()
    pl_long = agg.aggregate("Target_Long")
    pl_short = agg.aggregate("Target_Short")

    agg.save_pl(pl_long, pl_short)
