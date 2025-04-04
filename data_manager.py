import datetime
import json
import os
import pickle
import sqlite3

import pandas as pd
import requests
import yfinance as yf
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

from misc import Misc


class DataManager:
    def __init__(self, n_symbols=50):
        self.n_symbols = n_symbols
        self.data = []
        for _ in range(n_symbols):
            self.data.append([])

        load_dotenv()
        self.base_dir = os.getenv("BaseDir")
        self.db = f"{self.base_dir}/stock_database.db"

        self.base_url = "https://api.jquants.com/v1"

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

            dt_object = datetime.datetime.fromisoformat(
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
        now = datetime.datetime.now()
        filename = now.strftime("data_%Y%m%d_%H%M%S.pkl")

        dirname = f"{self.base_dir}/data"
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        filename = os.path.join(dirname, filename)

        with open(filename, "wb") as f:
            pickle.dump(data_list, f)

        return filename

    def set_token(self):
        """
        JPXのAPIを使うためのトークンを取得する
        """

        # .envファイルから環境変数を読み込む
        load_dotenv()
        email = os.getenv("Email")
        password = os.getenv("JPXPassword")

        # リフレッシュトークンを得る
        data = {"mailaddress": f"{email}", "password": f"{password}"}
        r_post = requests.post(
            self.base_url + "/token/auth_user", data=json.dumps(data)
        )
        self.refresh_token = r_post.json()["refreshToken"]

        # リフレッシュトークンを使ってIDトークンを得る
        r_post = requests.post(
            self.base_url + f"/token/auth_refresh?refreshtoken={self.refresh_token}"
        )
        self.id_token = r_post.json()["idToken"]

    def fetch_stock_list(self):
        """
        上場銘柄のリストを取得する
        """

        headers = {"Authorization": "Bearer {}".format(self.id_token)}
        r = requests.get(self.base_url + "/listed/info", headers=headers)
        list_stocks = r.json()["info"]

        return list_stocks

    def update_stock_data(self):
        """
        最新の株価データを取得し、SQLiteに保存する
        """

        list_stocks = self.fetch_stock_list()
        list_codes = []

        today = datetime.date.today().strftime("%Y%m%d")
        ago = (datetime.date.today() - relativedelta(years=3)).strftime("%Y%m%d")

        for stock in list_stocks:
            # 市場区分が「TOKYO PRO MARKET」または「その他」である銘柄を除外する
            if stock["MarketCode"] == "0109" or stock["MarketCode"] == "0105":
                continue

            code = stock["Code"][:-1]
            ep = f"/prices/daily_quotes?code={code}&from={ago}&to={today}"

            headers = {"Authorization": "Bearer {}".format(self.id_token)}
            r = requests.get(self.base_url + ep, headers=headers)
            prices = r.json()["daily_quotes"]

            list_prices = []

            for price in prices:
                if price:
                    list_prices.append(
                        [
                            price["Date"],
                            price["Open"],
                            price["High"],
                            price["Low"],
                            price["Close"],
                            price["Volume"],
                        ]
                    )
            df_prices = pd.DataFrame(
                list_prices,
                columns=["date", "open", "high", "low", "close", "volume"],
            )

            # 直近300日間の出来高の平均が50,000未満の銘柄を除外する
            if df_prices["volume"].tail(300).mean() < 50000:
                continue

            # データが少ない銘柄を除外する
            if len(df_prices) < 200:
                continue

            conn = sqlite3.connect(self.db)
            with conn:
                df_prices.to_sql(code, conn, if_exists="replace", index=False)

            list_codes.append(
                [
                    stock["Date"],
                    code,
                    stock["CompanyName"],
                    stock["MarketCodeName"],
                ]
            )

        df_codes = pd.DataFrame(list_codes, columns=["date", "code", "brand", "market"])

        conn = sqlite3.connect(self.db)
        with conn:
            df_codes.to_sql("Codes", conn, if_exists="replace", index=False)

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

        conn = sqlite3.connect(self.db)
        with conn:
            stocks_df.to_sql("Codes", conn, if_exists="replace", index=False)

        for code in stocks_df["code"]:
            try:
                data_df = yf.download(code + ".T", period="max", progress=False)
            except Exception:
                # なぜかたまにデータが取得できないことがあるので、その場合は削除・スキップする
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

        conn = sqlite3.connect(self.db)
        with conn:
            df = pd.read_sql_query(query, conn)

        return df

    def load_stock_list(self):
        conn = sqlite3.connect(self.db)
        with conn:
            df = pd.read_sql_query("select * from Codes;", conn)

        return df

    def fetch_target(self, table_name="Target", target_date="today"):
        if target_date == "today":
            target_date = datetime.datetime.now().strftime("%Y-%m-%d")

        conn = sqlite3.connect(self.db)
        with conn:
            df = pd.read_sql_query(
                f"select distinct * from {table_name} where date = ?;",
                conn,
                params=[target_date],
            )

        return df.values.tolist()

    def save_model_names(self, data_df):
        conn = sqlite3.connect(self.db)
        with conn:
            data_df.to_sql("Models", conn, if_exists="replace", index=False)

    def load_model_list(self):
        conn = sqlite3.connect(self.db)
        with conn:
            df = pd.read_sql_query("select * from Models;", conn)

        return df

    def save_order(self, data_df):
        conn = sqlite3.connect(self.db)
        with conn:
            data_df.to_sql("Orders", conn, if_exists="append", index=False)

    def load_order(self):
        conn = sqlite3.connect(self.db)
        with conn:
            df = pd.read_sql_query("select * from Orders;", conn)

        return df

    def update_price(self, order_id, price):
        conn = sqlite3.connect(self.db)
        with conn:
            conn.execute(
                f"UPDATE Orders SET price = {price} WHERE order_id = '{order_id}';",
            )

    def save_profit_loss(self, df, table_name="ProfitLoss"):
        conn = sqlite3.connect(self.db)
        with conn:
            df.to_sql(table_name, conn, if_exists="append", index=False)

    def seek_position(self, symbol, side):
        conn = sqlite3.connect(self.db)

        # 以下のクエリを実行して、指定した条件に一致する注文データを取得
        with conn:
            df = pd.read_sql_query(
                f"""
                SELECT *
                FROM Orders
                WHERE DATE(datetime) = date('now', 'localtime')
                AND symbol = {symbol}
                AND side = {str(side)};
                """,
                conn,
            )
        return df

    def calc_profitloss(self):
        df = self.load_order()

        today = datetime.datetime.now().strftime("%Y-%m-%d")
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        df = df[df["DateTime"].dt.date == pd.to_datetime(today).date()]

        df["Value"] = df["Price"] * df["Count"]
        result = (
            df.groupby(["Symbol", "Displayname", "Side"])["Value"]
            .sum()
            .unstack(fill_value=0)
        )
        result["Result"] = result["1"] - result["2"]
        result = result[["Result"]].reset_index().sort_values("Symbol")

        result["open"] = (
            df[df["Side"] == "2"].sort_values("Symbol")["Price"].reset_index(drop=True)
        )
        result["close"] = (
            df[df["Side"] == "1"].sort_values("Symbol")["Price"].reset_index(drop=True)
        )
        result["Count"] = (
            df[df["Side"] == "1"].sort_values("Symbol")["Count"].reset_index(drop=True)
        )

        result = result[["Symbol", "Displayname", "open", "close", "Count", "Result"]]
        return result


if __name__ == "__main__":
    # 土日祝日は実行しない
    misc = Misc()
    if misc.check_day_type(datetime.date.today()):
        exit()

    dm = DataManager()
    dm.set_token()
    dm.update_stock_data()
