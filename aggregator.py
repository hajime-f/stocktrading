import datetime
import sqlite3

from data_manager import DataManager
from misc import Misc


if __name__ == "__main__":
    if Misc().check_day_type(datetime.date.today()):
        exit()

    dm = DataManager()
    today = datetime.date.today().strftime("%Y-%m-%d")

    df_long = dm.load_table_by_date("Target_Long", today)
    pl_long = 0

    for index, row in df_long.iterrows():
        oc_prices = dm.load_open_close_prices(row["code"], today)
        pl_long += oc_prices["close"] - oc_prices["open"]

    df_short = dm.load_table_by_date("Target_Short", today)
    pl_short = 0

    for index, row in df_short.iterrows():
        oc_prices = dm.load_open_close_prices(row["code"], today)
        pl_short += oc_prices["open"] - oc_prices["close"]

    # conn = sqlite3.connect(dm.db)
    # with conn:
    #     conn.execute(
    #         f""""
    #         UPDATE Aggregate SET pl_long = {pl_long}, pl_short = {pl_short}
    #         WHERE date = '{today}';
    #         """,
    #     )
