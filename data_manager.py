import datetime
import json
import os
import pickle
import sqlite3
import threading

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
        self.thread_local = threading.local()

        self.base_url = "https://api.jquants.com/v1"

    def close(self):
        if hasattr(self.thread_local, "conn"):
            self.thread_local.conn.close()
            del self.thread_local.conn

    def _get_connection(self):
        # 今のスレッドにconn属性があるかチェック
        if not hasattr(self.thread_local, "conn"):
            # なければ、このスレッド専用の接続を作成
            self.thread_local.conn = sqlite3.connect(self.db)
        return self.thread_local.conn

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

        conn = self._get_connection()
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

        with conn:
            df_codes.to_sql("Codes", conn, if_exists="replace", index=False)

    def init_stock_data(self):
        """
        Yahoo!ファイナンスから日足の株価データを取得し、SQLiteデータベースに保存する
        """
        conn = self._get_connection()
        df_stocks = pd.DataFrame(self.fetch_stock_list())
        list_codes = []

        # 市場区分が「TOKYO PRO MARKET」または「その他」である銘柄を除外する
        df_stocks = df_stocks[df_stocks["MarketCode"] != "0109"]
        df_stocks = df_stocks[df_stocks["MarketCode"] != "0105"]

        for index, df in df_stocks.iterrows():
            code = df["Code"][:-1]
            try:
                df_prices = yf.download(
                    code + ".T", period="2y", auto_adjust=True, progress=False
                )
            except Exception:
                continue

            # データが少ない銘柄を除外する
            if len(df_prices) < 200:
                continue

            df_prices.columns = df_prices.columns.get_level_values(0)
            df_prices.columns = df_prices.columns.str.lower()
            df_prices["date"] = df_prices.index
            df_prices = df_prices.reindex(
                columns=["date", "open", "high", "low", "close", "volume"]
            )
            df_prices = df_prices.reset_index(drop=True)
            df_prices["date"] = pd.to_datetime(df_prices["date"]).dt.strftime(
                "%Y-%m-%d"
            )

            # 直近300日間の出来高の平均が50,000未満の銘柄を除外する
            if df_prices["volume"].tail(300).mean() < 50000:
                continue

            with conn:
                df_prices.to_sql(code, conn, if_exists="replace", index=False)

            list_codes.append(
                [df["Date"], code, df["CompanyName"], df["MarketCodeName"]]
            )

        df_codes = pd.DataFrame(list_codes, columns=["date", "code", "brand", "market"])

        with conn:
            df_codes.to_sql("Codes", conn, if_exists="replace", index=False)

    def load_stock_data(self, code, start="start", end="end"):
        conn = self._get_connection()

        if start == "start" and end == "end":
            query = f'select * from "{code}" order by date;'
        elif start == "start":
            query = f'select * from "{code}" where date <= "{end}" order by date;'
        elif end == "end":
            query = f'select * from "{code}" where date >= "{start}" order by date;'
        else:
            query = f'select * from "{code}" where date >= "{start}" and date <= "{end}" order by date;'

        try:
            with conn:
                df = pd.read_sql_query(query, conn)
        except pd.errors.DatabaseError:
            df = pd.DataFrame()

        return df

    def load_stock_list(self):
        conn = self._get_connection()
        with conn:
            df = pd.read_sql_query("select * from Codes;", conn)

        return df

    def fetch_target(self, table_name="Target", target_date="today"):
        conn = self._get_connection()

        if target_date == "today":
            target_date = datetime.datetime.now().strftime("%Y-%m-%d")
        sql_query = f"select distinct * from '{table_name}' where date = ?;"

        with conn:
            try:
                df = pd.read_sql_query(sql_query, conn, params=[target_date])
            except pd.errors.DatabaseError as e:
                self.logger.error(f"データベースエラー: {e}")
                df = pd.DataFrame()

        return df

    def save_model_names(self, data_df):
        conn = self._get_connection()
        with conn:
            data_df.to_sql("Models", conn, if_exists="replace", index=False)

    def load_model_list(self):
        conn = self._get_connection()
        with conn:
            df = pd.read_sql_query("select * from Models;", conn)

        return df

    def load_aggregate(self):
        conn = self._get_connection()
        with conn:
            df = pd.read_sql_query("select * from Aggregate;", conn)

        return df

    def save_order(self, data_df):
        conn = self._get_connection()
        with conn:
            data_df.to_sql("Orders", conn, if_exists="append", index=False)

    def load_order(self, table_name="Orders", target_date="today"):
        conn = self._get_connection()

        if target_date == "today":
            target_date = datetime.datetime.now().strftime("%Y-%m-%d")

        with conn:
            sql_query = f"select distinct * from '{table_name}' where date = ?;"
            df = pd.read_sql_query(sql_query, conn, params=[target_date])

        return df

    def update_price(self, order_id, price):
        conn = self._get_connection()
        with conn:
            sql_query = "update Orders set price = ? where order_id = ?;"
            conn.execute(sql_query, params=[price, order_id])

    def save_profit_loss(self, df, table_name="ProfitLoss"):
        conn = self._get_connection()
        with conn:
            df.to_sql(table_name, conn, if_exists="append", index=False)

    def seek_position(self, symbol, side):
        conn = self._get_connection()

        # 以下のクエリを実行して、指定した条件に一致する注文データを取得
        with conn:
            sql_query = """
            select *
            from Orders
            where DATE(datetime) = date('now', 'localtime')
            and symbol = ?
            and side = ?;
            """
            df = pd.read_sql_query(sql_query, conn, params=[symbol, side])
        return df

    def find_newest_close_price(self, symbol):
        conn = self._get_connection()
        with conn:
            sql_query = f"select close from '{symbol}' order by date desc limit 1;"
            df = pd.read_sql_query(sql_query, conn)
        return df["close"].item()

    def load_table_by_date(self, table_name, date):
        conn = self._get_connection()
        with conn:
            sql_query = f"select * from '{table_name}' where date = ?;"
            df = pd.read_sql_query(sql_query, conn, params=[date])

        return df

    def load_open_close_prices(self, code, date):
        conn = self._get_connection()
        with conn:
            sql_query = f"select open, close from '{code}' where date = ?;"
            df = pd.read_sql_query(sql_query, conn, params=[date])

        return df

    def check_stock_data(self, code, val_date):
        conn = self._get_connection()
        with conn:
            sql_query = f"select * from '{code}' order by date desc limit 1;"
            df = pd.read_sql_query(sql_query, conn)

        if df.empty:
            return False

        last_date = df["date"].item()
        if last_date == val_date:
            return True
        else:
            return False

    def save_execution(self, df_data):
        conn = self._get_connection()
        with conn:
            df_data.to_sql("Execution", conn, if_exists="append", index=False)

    def load_execution(self, order_id):
        conn = self._get_connection()
        with conn:
            df = pd.read_sql_query(
                "select * from Execution where order_id = ?;",
                conn,
                params=[order_id],
            )

        return df

    def seek_execution(self, symbol, side):
        conn = self._get_connection()

        # 以下のクエリを実行して、指定した条件に一致する注文データを取得
        with conn:
            sql_query = """
                select *
                from Execution
                where DATE(exec_time) = date('now', 'localtime')
                and symbol = ?
                and side = ?;
            """
            df = pd.read_sql_query(sql_query, conn, params=[symbol, side])

        return df

    def save_result(self, df_result):
        conn = self._get_connection()

        with conn:
            df_result.to_sql("Result", conn, if_exists="append", index=False)


if __name__ == "__main__":
    # 土日祝日は実行しない
    misc = Misc()
    if misc.check_day_type(datetime.date.today()):
        exit()

    dm = DataManager()
    dm.set_token()
    dm.init_stock_data()
