import os
import pickle
from datetime import datetime

import yfinance as yf
import sqlite3
import pandas as pd


class DataManagement:
    def __init__(self, n_symbols=50):
        self.n_symbols = n_symbols
        self.data = []
        for _ in range(n_symbols):
            self.data.append([])

    def append_data(self, new_data, index):
        if new_data["CurrentPriceTime"] is not None:
            data = {
                "CurrentPriceTime": new_data["CurrentPriceTime"],
                "CurrentPrice": new_data["CurrentPrice"],
                "TradingVolume": new_data["TradingVolume"],
            }
            self.data[index].append(data)

    def prepare_dataframe_list(self, symbols):
        # 生データを分足のDataFrameに変換する
        df_list = [self.convert_to_dataframe(d) for d in self.data]
        return df_list

    def convert_to_dataframe(self, original_data):
        price_list = []

        for d in original_data:
            if d["CurrentPriceTime"] is None:
                continue

            dt_object = datetime.fromisoformat(
                d["CurrentPriceTime"].replace("Z", "+00:00")
            )
            formatted_datetime = dt_object.strftime("%Y-%m-%d %H:%M")
            price_list.append([formatted_datetime, d["CurrentPrice"]])

        price_df = pd.DataFrame(price_list, columns=["DateTime", "Price"])
        price_df = price_df.set_index("DateTime")
        price_df.index = pd.to_datetime(price_df.index)
        price_df = price_df.resample("1Min").ohlc().dropna()
        price_df.columns = price_df.columns.get_level_values(1)

        return price_df

    def save_data(self, data_list):
        now = datetime.now()
        filename = now.strftime("data_%Y%m%d_%H%M%S.pkl")

        dirname = "./data"
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        filename = os.path.join(dirname, filename)

        with open(filename, "wb") as f:
            pickle.dump(data_list, f)

        return filename

    def init_stock_data(self):
        """
        Yahoo!ファイナンスから日足の株価データを取得し、SQLiteデータベースに保存する
        """
        stocks_df = pd.read_csv("./data/data_j.csv")

        conn = sqlite3.connect("./data/stock_data.db")
        with conn:
            stocks_df.to_sql("Codes", conn, if_exists="replace", index=False)

        for code in stocks_df["code"]:
            data_df = yf.download(code + ".T", start="2020-01-01", end=datetime.now())
            # data_df = yf.download(code + ".T", period="max")
            data_df.columns = data_df.columns.get_level_values(0)
            data_df.columns = data_df.columns.str.lower()
            data_df["date"] = data_df.index
            data_df = data_df.reindex(
                columns=["date", "open", "high", "low", "close", "volume"]
            )
            data_df = data_df.reset_index(drop=True)
            data_df["date"] = pd.to_datetime(data_df["date"]).dt.strftime("%Y-%m-%d")

            conn = sqlite3.connect("./data/stock_data.db")
            with conn:
                data_df.to_sql(code, conn, if_exists="replace", index=False)

    def load_stock_data(self, code):
        query = f'select * from "{code}" order by date;'

        conn = sqlite3.connect("./data/stock_data.db")
        with conn:
            df = pd.read_sql_query(query, conn)

        return df

    def load_stock_list(self):
        conn = sqlite3.connect("./data/stock_data.db")
        with conn:
            df = pd.read_sql_query("select * from Codes;", conn)

        return df


if __name__ == "__main__":
    dm = DataManagement()
    dm.init_stock_data()
