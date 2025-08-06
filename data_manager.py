import datetime
import json
import os
import sqlite3
import threading

import pandas as pd
import requests
import yfinance as yf

from exception import APIError
from config_manager import cm
from misc import Misc


class DataManager:
    def __init__(self):
        self.base_dir = cm.get("directory.base_dir")
        self.db = f"{self.base_dir}/{cm.get('database.name')}"
        self.thread_local = threading.local()

        self.base_url = cm.get("api.jpx.base_url")

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

    def set_token(self):
        """
        JPXのAPIを使うためのトークンを取得する
        """

        # .envファイルから環境変数を読み込む
        email = cm.get("api.jpx.email")
        password = cm.get("api.jpx.password")

        # リフレッシュトークンを得る
        data = {"mailaddress": f"{email}", "password": f"{password}"}

        try:
            r_post = requests.post(
                self.base_url + "/token/auth_user", data=json.dumps(data), timeout=10
            )
            r_post.raise_for_status()
            self.refresh_token = r_post.json()["refreshToken"]

            # リフレッシュトークンを使ってIDトークンを得る
            r_post_id = requests.post(
                self.base_url
                + f"/token/auth_refresh?refreshtoken={self.refresh_token}",
                timeout=10,
            )
            r_post_id.raise_for_status()
            self.id_token = r_post_id.json()["idToken"]

        except requests.exceptions.RequestException as e:
            # ネットワークエラー、タイムアウト、HTTPエラーステータスなどをまとめて捕捉
            self.logger.critical(f"Failed to communicate with J-QUANTS API: {e}")
            raise APIError("J-QUANTS APIとの通信に失敗しました。") from e
        except KeyError as e:
            # レスポンスに期待したキーがなかった場合
            self.logger.critical(
                f"Unexpected API response format from J-QUANTS: missing key {e}"
            )
            raise APIError(
                "J-QUANTS APIからのレスポンスのフォーマットが不正です。"
            ) from e

    def fetch_stock_list(self):
        """
        上場銘柄のリストを取得する
        """

        headers = {"Authorization": "Bearer {}".format(self.id_token)}
        r = requests.get(self.base_url + "/listed/info", headers=headers)
        list_stocks = r.json()["info"]

        return list_stocks

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
                    code + ".T", period="5y", auto_adjust=True, progress=False
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
            df = pd.read_sql_query(sql_query, conn, params=[target_date])

        return df

    def save_order(self, data_df):
        conn = self._get_connection()
        with conn:
            data_df.to_sql("Orders", conn, if_exists="append", index=False)

    def load_order(self):
        conn = self._get_connection()

        with conn:
            sql_query = f"select distinct * from Orders where DATE(datetime) = date('now', 'localtime');"
            df = pd.read_sql_query(sql_query, conn)

        return df

    def delete_order(self, order_id):
        conn = self._get_connection()
        with conn:
            sql_query = "delete from Orders where order_id = ?;"
            conn.execute(sql_query, params=[order_id])

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

    def get_brand(self, symbol):
        conn = self._get_connection()

        with conn:
            sql_query = f"SELECT brand FROM Codes WHERE code = '{symbol}';"
            df = pd.read_sql_query(sql_query, conn).head(1)

        return df["brand"].item() if not df.empty else None

    def execute_query(self, query):
        conn = self._get_connection()

        with conn:
            cursor = conn.cursor()
            cursor.execute(query)


if __name__ == "__main__":
    # 土日祝日は実行しない
    if Misc.check_day_type(datetime.date.today()):
        exit()

    dm = DataManager()

    # Executionテーブルがない場合は作る
    create_sql = """
    CREATE TABLE IF NOT EXISTS Execution (
        exec_time TEXT NOT NULL,
        recv_time TEXT NOT NULL,
        symbol TEXT NOT NULL,
        displayname TEXT NOT NULL,
        price REAL NOT NULL,
        qty INTEGER NOT NULL,
        order_id TEXT NOT NULL,
        execution_id TEXT PRIMARY KEY,
        side INTEGER NOT NULL
    );
    
    """
    dm.execute_query(create_sql)

    # Ordersテーブルがない場合は作る
    create_sql = """
    CREATE TABLE IF NOT EXISTS Orders (
        datetime TEXT NOT NULL,
        symbol TEXT NOT NULL,
        displayname TEXT NOT NULL,
        price REAL,
        qty INTEGER NOT NULL,
        order_id TEXT PRIMARY KEY,
        side INTEGER NOT NULL
    );
    
    """
    dm.execute_query(create_sql)

    # logsディレクトリがない場合は作る
    os.makedirs("logs", exist_ok=True)

    dm.set_token()
    dm.init_stock_data()
