import datetime as dt
import yfinance as yf
import sqlite3
import pandas as pd


class DataManagement:
    def __init__(self):
        pass

    def init_stock_data(self):
        stocks_df = pd.read_csv("./data/data_j.csv")

        conn = sqlite3.connect("./data/stock_data.db")
        with conn:
            stocks_df.to_sql("Codes", conn, if_exists="replace", index=False)

        for code in stocks_df["code"]:
            data_df = yf.download(code + ".T", start="2021-01-01", end=dt.date.today())
            data_df.columns = data_df.columns.get_level_values(0)
            data_df.columns = data_df.columns.str.lower()
            data_df = data_df.reindex(
                columns=["open", "high", "low", "close", "volume"]
            )

            conn = sqlite3.connect("./data/stock_data.db")
            with conn:
                data_df.to_sql(code, conn, if_exists="replace")

    def load_stock_data(self, code):
        query = f"select distinct date from {code} order by date;"

        conn = sqlite3.connect("./data/stock_data.db")
        with conn:
            df = pd.read_sql_query(query, conn)

        return df


if __name__ == "__main__":
    dm = DataManagement()
    dm.init_stock_data()
