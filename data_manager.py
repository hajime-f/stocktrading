import os
import pickle
import sqlite3
from datetime import datetime

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv


class DataManager:
    def __init__(self, n_symbols=50):
        self.n_symbols = n_symbols
        self.data = []
        for _ in range(n_symbols):
            self.data.append([])

        load_dotenv()
        self.base_dir = os.getenv("BaseDir")

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

        dirname = f"{self.base_dir}/data"
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
        stocks_df = pd.read_csv(f"{self.base_dir}/data/data_j.csv")

        # market が「ETF・ETN」「PRO Market」「REIT」「出資証券」は削除する
        stocks_df = stocks_df[stocks_df["market"] != "ETF・ETN"]
        stocks_df = stocks_df[stocks_df["market"] != "PRO Market"]
        stocks_df = stocks_df[
            stocks_df["market"]
            != "REIT・ベンチャーファンド・カントリーファンド・インフラファンド"
        ]
        stocks_df = stocks_df[stocks_df["market"] != "出資証券"]

        conn = sqlite3.connect(f"{self.base_dir}/data/stock_data.db")
        with conn:
            stocks_df.to_sql("Codes", conn, if_exists="replace", index=False)

        for code in stocks_df["code"]:
            data_df = yf.download(code + ".T", period="max", progress=False)

            # なぜかたまにデータが取得できないことがあるので、その場合は削除・スキップする
            if data_df.empty:
                with conn:
                    conn.execute(f"delete from Codes where code = '{code}';")
                continue

            # データの少ない銘柄は削除・スキップする
            if len(data_df) < 100:
                with conn:
                    conn.execute(f"delete from Codes where code = '{code}';")
                continue

            data_df.columns = data_df.columns.get_level_values(0)
            data_df.columns = data_df.columns.str.lower()
            data_df["date"] = data_df.index
            data_df = data_df.reindex(
                columns=["date", "open", "high", "low", "close", "volume"]
            )
            data_df = data_df.reset_index(drop=True)
            data_df["date"] = pd.to_datetime(data_df["date"]).dt.strftime("%Y-%m-%d")

            # 最近の出来高が小さい銘柄は削除・スキップする
            if data_df["volume"].tail(500).mean() < 50000:
                with conn:
                    conn.execute(f"delete from Codes where code = '{code}';")
                continue

            with conn:
                data_df.to_sql(code, conn, if_exists="replace", index=False)

    def load_stock_data(self, code, start="start", end="end"):
        if start == "start" and end == "end":
            query = f'select * from "{code}" order by date;'
        elif start == "start":
            query = f'select * from "{code}" where date <= "{end}" order by date;'
        elif end == "end":
            query = f'select * from "{code}" where date >= "{start}" order by date;'
        else:
            query = f'select * from "{code}" where date >= "{start}" and date <= "{end}" order by date;'

        conn = sqlite3.connect(f"{self.base_dir}/data/stock_data.db")
        with conn:
            df = pd.read_sql_query(query, conn)

        return df

    def load_stock_list(self):
        conn = sqlite3.connect(f"{self.base_dir}/data/stock_data.db")
        with conn:
            df = pd.read_sql_query("select * from Codes;", conn)

        return df

    def fetch_target(self, target_date="today"):
        if target_date == "today":
            target_date = datetime.now().strftime("%Y-%m-%d")

        conn = sqlite3.connect(f"{self.base_dir}/data/stock_data.db")
        with conn:
            df = pd.read_sql_query(
                "select distinct * from Target where date = ?;",
                conn,
                params=[target_date],
            )

        return df.values.tolist()

    def save_model_names(self, data_df):
        conn = sqlite3.connect(f"{self.base_dir}/data/stock_data.db")
        with conn:
            data_df.to_sql("Models", conn, if_exists="replace", index=False)

    def load_model_list(self):
        conn = sqlite3.connect(f"{self.base_dir}/data/stock_data.db")
        with conn:
            df = pd.read_sql_query("select * from Models;", conn)

        return df

    def save_order(self, data_df):
        conn = sqlite3.connect(f"{self.base_dir}/data/stock_data.db")
        with conn:
            data_df.to_sql("Orders", conn, if_exists="append", index=False)

    def load_order(self):
        conn = sqlite3.connect(f"{self.base_dir}/data/stock_data.db")
        with conn:
            df = pd.read_sql_query("select * from Orders;", conn)

        return df

    def update_price(self, order_id, price):
        conn = sqlite3.connect(f"{self.base_dir}/data/stock_data.db")
        with conn:
            conn.execute(
                f"UPDATE Orders SET Price = {price} WHERE Order_id = {order_id};",
            )

    def save_profit_loss(self, df):
        conn = sqlite3.connect(f"{self.base_dir}/data/stock_data.db")
        with conn:
            df.to_sql("ProfitLoss", conn, if_exists="append", index=False)

    def seek_position(self, symbol, side):
        conn = sqlite3.connect(f"{self.base_dir}/data/stock_data.db")

        # 以下のクエリを実行して、指定した条件に一致する注文データを取得
        with conn:
            df = pd.read_sql_query(
                f"""
                SELECT *
                FROM Orders
                WHERE DATE(DateTime) = date('now', 'localtime')
                AND Symbol = {symbol}
                AND Side = {str(side)};
                """,
                conn,
            )
        return df

    def calculate_price_diff_times_count(self):
        """本日日付のデータから、Side1のPrice - Side2のPrice * Count を Symbol ごとに計算する."""
        df = self.load_order()

        today = datetime.now().strftime("%Y-%m-%d")
        df["DateTime"] = pd.to_datetime(df["DateTime"])

        df_today = df[df["DateTime"].dt.date == pd.to_datetime(today).date()]
        result = {}

        def calculate_diff_times_count(x):
            side_1_price = (
                x[x["Side"] == 1]["Price"].iloc[0]
                if not x[x["Side"] == 1].empty
                else None
            )
            side_2_price = (
                x[x["Side"] == 2]["Price"].iloc[0]
                if not x[x["Side"] == 2].empty
                else None
            )
            side_1_count = (
                x[x["Side"] == 1]["Count"].iloc[0]
                if not x[x["Side"] == 1].empty
                else None
            )
            side_2_count = (
                x[x["Side"] == 2]["Count"].iloc[0]
                if not x[x["Side"] == 2].empty
                else None
            )

            if (
                side_1_price is not None
                and side_2_price is not None
                and side_1_count is not None
                and side_2_count is not None
            ):
                return (side_1_price - side_2_price) * side_1_count
            else:
                return None

        result = df_today.groupby("Symbol").apply(calculate_diff_times_count).to_dict()
        return result


if __name__ == "__main__":
    dm = DataManager()
    dm.init_stock_data()
