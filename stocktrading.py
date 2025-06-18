import os
import random
import signal
import sys
import threading
import time
from datetime import date, datetime
from logging import config, getLogger
from typing import Dict

import pandas as pd
import yaml
from dotenv import load_dotenv

from data_manager import DataManager
from exception import APIError, ConfigurationError, DataProcessingError
from library import Library
from misc import MessageManager, Misc
from stock import Stock


class StockTrading:
    def __init__(self):
        # 定数の定義
        self.POLLING_INTERVAL = 300  # ポーリング間隔 (秒)
        self.POLLING_INTERVAL_VARIATION = 30  # ポーリング間隔の変動幅 (秒)

        # スレッドを停止させるためのイベント
        self.stop_event = threading.Event()

        # 銘柄データを保持する辞書
        self.stocks: Dict[str, Stock] = {}

        # 損益を保持する辞書
        self.profit_loss: Dict[str, list] = {}
        self.profit_loss_lock = threading.Lock()

        # メッセージマネージャーのインスタンス化
        self.msg = MessageManager()

        # ロガーの初期化
        self.logger = getLogger(__name__)
        self._init_logger()

        # 株ライブラリを初期化
        self.lib = Library()

        # データマネージャーを初期化
        self.dm = DataManager()

    def _init_logger(self):
        # ロガーを初期化
        load_dotenv()
        path_name = os.getenv("BaseDir")
        file_name = os.getenv("LogConfigFile")
        if file_name is None:
            self.logger.critical(self.msg.get("errors.env_not_found", env_file=".env"))
            sys.exit(1)
        log_conf_file = f"{path_name}/{file_name}"
        try:
            with open(log_conf_file, "rt") as f:
                config.dictConfig(yaml.safe_load(f.read()))
        except FileNotFoundError:
            self.logger.critical(self.msg.get("errors.file_not_found", path=path_name))
            sys.exit(1)

    def setup_environment(self):
        today = date.today().strftime("%Y年%m月%d日")
        self.logger.info(self.msg.get("info.program_start", today=today))

        # 銘柄を登録する
        self.register_stocks()

        # 取引余力を取得
        wallet_cash = f"{int(self.lib.wallet_cash()):,}"
        self.logger.info(self.msg.get("info.wallet_cash", wallet_cash=wallet_cash))

        # 受信関数を登録
        self.lib.register_receiver(self.receive)

        # Ctrl+C ハンドラーを登録
        signal.signal(signal.SIGINT, self.signal_handler)

    def register_stocks(self):
        # 今回取引する銘柄リストを取得
        # target_stocks = dm.fetch_target()
        target_symbols = [
            ["2025-06-02", "1475", "iシェアーズ・コア TOPIX ETF", 0.999, 1],
        ]
        columns = ["date", "code", "brand", "pred", "side"]
        target_stocks = pd.DataFrame(target_symbols, columns=columns)

        try:
            # 登録銘柄リストからすべての銘柄をいったん削除する
            self.lib.unregister_all()

            # 銘柄登録
            self.lib.register(target_stocks["code"].tolist())

        except APIError as e:
            self.logger.critical(e)
            sys.exit(1)

        # Stockクラスをインスタンス化して辞書に入れる
        for _, row in target_stocks.iterrows():
            symbol = row["code"]
            stock_instance = Stock(symbol, self.lib, self.dm, row["side"], row["brand"])
            stock_instance.set_information()
            self.stocks[symbol] = stock_instance

    def prepare_threads(self):
        # スレッドを準備
        threads = [
            threading.Thread(target=self.run_polling, args=(st,))
            for st in self.stocks.values()
        ]
        push_receiver_thread = threading.Thread(
            target=self.lib.run, args=(self.stop_event,), daemon=True
        )

        # スレッドを起動
        self.logger.info(self.msg.get("info.thread_starting"))
        for thread in threads:
            thread.start()
        push_receiver_thread.start()
        self.logger.info(self.msg.get("info.all_thread_started"))

        return threads

    def run_main_loop(self):
        # スレッドを準備
        threads = self.prepare_threads()

        while True:
            now = datetime.now()
            if now.hour > 15 or (now.hour == 15 and now.minute >= 35):
                self.logger.info(self.msg.get("info.time_terminate"))
                self.stop_event.set()

            # 停止イベントがセットされたら、監視ループを抜ける（Ctrl+Cまたは時間経過）
            if self.stop_event.is_set():
                break

            # 10秒ごとにチェック
            time.sleep(10)

        return threads

    # PUSH配信を受信した時に呼ばれる関数
    def receive(self, data: Dict):
        # 受信したデータに対応する銘柄を取得する
        symbol = data.get("Symbol")

        # データを追加する
        if symbol in self.stocks:
            self.stocks[symbol].append_data(data)

    # 約５分間隔でstockクラスのpolling関数を呼ぶ関数
    def run_polling(self, st):
        try:
            self.logger.info(
                self.msg.get(
                    "info.transaction_start", symbol=st.symbol, disp_name=st.disp_name
                )
            )
            last_polling_time = time.time()

            while not self.stop_event.is_set():
                if time.time() - last_polling_time >= self.POLLING_INTERVAL:
                    time.sleep(random.uniform(0, self.POLLING_INTERVAL_VARIATION))

                    if self.stop_event.is_set():
                        break

                    st.polling()
                    last_polling_time = time.time()

                time.sleep(1)

            # while文を抜けたときに実行する処理
            if st.check_transaction():
                sell_price, buy_price = st.fetch_prices()
                with self.profit_loss_lock:
                    self.profit_loss[st.symbol] = [
                        st.disp_name,
                        st.symbol,
                        sell_price,
                        buy_price,
                        st.side,
                    ]

        except Exception as e:
            self.logger.critical(
                self.msg.get(
                    "errors.polling_thread_error",
                    thread_name=threading.current_thread().name,
                    reason=e,
                ),
                exc_info=True,
            )
            self.stop_event.set()

        finally:
            st.dm.close()

    # Ctrl+C ハンドラー
    def signal_handler(self, sig, frame):
        self.logger.warning(self.msg.get("info.terminate"))
        self.stop_event.set()  # スレッド停止イベントを設定

    def display_profitloss(self):
        # 損益を表示する
        pl_sum = 0
        list_result = []
        today = date.today().strftime("%Y-%m-%d")

        with self.profit_loss_lock:
            profit_loss_copy = self.profit_loss.copy()

        breakpoint()

        self.logger.info("--- 損益計算結果 ---")
        for pl in profit_loss_copy.values():
            if pl[2] is not None and pl[3] is not None:
                diff = pl[2] - pl[3]
                self.logger.info(
                    f"[yellow]{pl[0]}[/] ({pl[1]}): 売値 = {pl[2]:,.0f} 円, 買値 = {pl[3]:,.0f} 円: 損益 = {diff:,.0f} 円"
                )
                pl_sum += diff
                list_result.append([today, pl[0], pl[1], pl[2], pl[3], diff, pl[4]])
            else:
                self.logger.warning(
                    f"{pl[0]} ({pl[1]}): 売値・買値を特定できませんでした。"
                )
        self.logger.info("--------------------")
        self.logger.info(f"[red]合計損益[/]: {pl_sum:,.0f} 円")

        return pl_sum, list_result

    def process_profitloss(self):
        # 損益を表示する
        pl_sum, list_result = self.display_profitloss()

        # 損益を記録
        df_profit_loss = pd.DataFrame(
            list_result,
            columns=[
                "date",
                "brand",
                "symbol",
                "sell_price",
                "buy_price",
                "profit_loss",
                "side",
            ],
        )
        self.dm.save_profit_loss(df_profit_loss)

        wallet_cash = f"{int(self.lib.wallet_cash()):,}"
        result = pd.DataFrame(
            [date.today().strftime("%Y-%m-%d"), wallet_cash, pl_sum],
            columns=["date", "cash", "profit_loss"],
        )
        self.dm.save_result(result)

    def main(self):
        threads = []
        exit_code = 0

        try:
            self.setup_environment()

            # メインループ
            threads = self.run_main_loop()

            # 結果表示
            self.process_profitloss()

        except (ConfigurationError, APIError) as e:
            self.logger.critical(e)
            exit_code = 1

        except DataProcessingError as e:
            self.logger.critical(e)
            exit_code = 1

        except RuntimeError as e:
            self.logger.critical(self.msg.get("errors.thread_launch_failed", reason=e))
            exit_code = 1

        except Exception as e:
            self.logger.critical(
                self.msg.get("errors.thread_unexpected_error", reason=e), exc_info=True
            )
            exit_code = 1

        finally:
            self.stop_event.set()
            self.dm.close()

            # すべてのスレッドが終了するのを待つ
            for thread in threads:
                thread.join()

            self.logger.info(self.msg.get("info.program_end"))
            sys.exit(exit_code)


if __name__ == "__main__":
    # 土日祝日は実行しない
    if Misc().check_day_type(date.today()):
        print("本日は土日祝日です。プログラムを終了します。")
        sys.exit(0)

    stocktrading = StockTrading()
    stocktrading.main()
