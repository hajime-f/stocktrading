import os
import queue
import random
import signal
import sys
import threading
import time
from datetime import date, datetime
from logging import config, getLogger
from typing import Dict

import numpy as np
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
        self.POLLING_INTERVAL = 180  # ポーリング間隔 (秒)
        self.POLLING_INTERVAL_VARIATION = 30  # ポーリング間隔の変動幅 (秒)
        self.MONITOR_INTERVAL = 3600  # リアルタイム損益の監視間隔（秒）

        # スレッドを停止させるためのイベント
        self.stop_event = threading.Event()

        # 銘柄データを保持する辞書
        self.stocks: Dict[str, Stock] = {}

        # 損益を保持する辞書
        self.profit_loss: Dict[str, list] = {}

        # 取引量を読み込み
        load_dotenv()
        self.base_transaction = int(os.getenv("BaseTransaction"))

        # スレッド間のエラー通知用キュー
        self.error_queue = queue.Queue()

        # スレッド間の最終損益報告用キューを追加
        self.result_queue = queue.Queue()

        # ザラ場中のリアルタイム損益を監視するためのキュー
        self.realtime_queue = queue.Queue()

        # 各銘柄の最新の未実現損益を保持する辞書
        self.current_pl: Dict[str, float] = {}

        # メッセージマネージャーのインスタンス化
        self.msg = MessageManager()

        # ロガーの初期化
        self.logger = getLogger(__name__)
        self._init_logger()

        # 株ライブラリを初期化
        self.lib = Library()

        # データマネージャーを初期化
        self.dm = DataManager()

        # 終了コード
        self.exit_code = 0

        # 取引余力
        self.wallet_cash = ""

    def _init_logger(self):
        # ロガーを初期化
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

        # 取引余力を取得
        self.wallet_cash = f"{int(self.lib.wallet_cash()):,}"
        self.logger.info(self.msg.get("info.wallet_cash", wallet_cash=self.wallet_cash))

        # 銘柄を登録する
        self.register_stocks()

        # 受信関数を登録
        self.lib.register_receiver(self.receive)

        # Ctrl+C ハンドラーを登録
        signal.signal(signal.SIGINT, self.signal_handler)

    def register_stocks(self):
        # 今回取引する銘柄リストを取得
        target_stocks = self.dm.fetch_target()

        try:
            # 登録銘柄リストからすべての銘柄をいったん削除する
            self.lib.unregister_all()

            # 銘柄登録
            self.lib.register(target_stocks["code"].tolist())

        except APIError as e:
            self.logger.critical(e)
            sys.exit(1)

        total_risk_amount = self.base_transaction * int(os.getenv("AllowableRisk"))
        target_stocks = self.calc_risk_amount(target_stocks, total_risk_amount)

        # Stockクラスをインスタンス化して辞書に入れる
        for _, row in target_stocks.iterrows():
            symbol = row["code"]
            stock_instance = Stock(
                symbol,
                self.lib,
                self.dm,
                row["side"],
                row["brand"],
                row["risk_amount"],
                self.base_transaction,
                realtime_queue=self.realtime_queue,
            )
            stock_instance.set_information()
            self.stocks[symbol] = stock_instance

    def calc_risk_amount(self, target_stocks, total_risk_amount):
        weights = target_stocks["pred"].to_numpy()
        risk_ratio = weights / np.sum(weights)

        target_stocks["risk_amount"] = total_risk_amount * risk_ratio
        return target_stocks

    def prepare_threads(self):
        # スレッドを準備
        threads = [
            threading.Thread(
                target=self.run_polling, args=(st, self.error_queue, self.result_queue)
            )
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

        # 監視時間を初期化
        last_monitor_time = time.time()

        while True:
            # スレッドからのエラーがないかチェックする
            self.check_thread_errors()

            # ザラ場中の損益をモニタリング
            if time.time() - last_monitor_time >= self.MONITOR_INTERVAL:
                self.monitor_realtime_pl()
                last_monitor_time = time.time()

            now = datetime.now()
            if now.hour > 15 or (now.hour == 15 and now.minute >= 30):
                self.logger.info(self.msg.get("info.time_terminate"))
                self.stop_event.set()

            # 停止イベントがセットされたら、監視ループを抜ける（Ctrl+Cまたは時間経過）
            if self.stop_event.is_set():
                break

            # 10秒ごとにチェック
            time.sleep(10)

        return threads

    def check_thread_errors(self):
        # スレッドからのエラー報告キューをチェックし、エラーがあれば処理する
        try:
            # キューからエラー情報を取得
            error_info = self.error_queue.get_nowait()
            symbol = error_info["symbol"]
            exception = error_info["exception"]

            self.logger.error(
                self.msg.get(
                    "error.thread_critical_error", symbol=symbol, exception=exception
                )
            )

            # エラーが発生した銘柄をアクティブなリストから除く
            if symbol in self.stocks:
                del self.stocks[symbol]
            if symbol in self.current_pl:
                del self.current_pl[symbol]

        except queue.Empty:
            # キューが空の場合は何もしない
            pass

    def monitor_realtime_pl(self):
        # リアルタイム損益キューを処理し、途中経過を表示する
        while not self.realtime_queue.empty():
            try:
                update = self.realtime_queue.get_nowait()
                self.current_pl[update["symbol"]] = update["unrealized_pl"]
            except queue.Empty:
                break

        if not self.current_pl:
            return

        total_pl = sum(self.current_pl.values())
        self.logger.info(self.msg.get("info.realtime_pl", total_pl=f"{total_pl:,.0f}"))

    # PUSH配信を受信した時に呼ばれる関数
    def receive(self, data: Dict):
        # 受信したデータに対応する銘柄を取得する
        symbol = data.get("Symbol")

        # データを追加する
        if symbol in self.stocks:
            self.stocks[symbol].append_data(data)

    def process_final_pl_queue(self):
        # 最終損益キューを処理して、結果をself.profit_lossに格納
        while not self.result_queue.empty():
            try:
                pl_data = self.result_queue.get_nowait()
                symbol = pl_data["symbol"]
                self.profit_loss[symbol] = [
                    pl_data["disp_name"],
                    pl_data["symbol"],
                    pl_data["sell_price"],
                    pl_data["buy_price"],
                    pl_data["side"],
                ]
            except queue.Empty:
                break

    # 約５分間隔でstockクラスのpolling関数を呼ぶ関数
    def run_polling(self, st, error_queue, result_queue):
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

            # 念のため最後に一度 polling しておく
            if not self.exit_code:
                time.sleep(random.uniform(0, self.POLLING_INTERVAL_VARIATION))
                st.polling()

            if st.check_transaction():
                sell_price, buy_price = st.fetch_prices()
                pl_data = {
                    "disp_name": st.disp_name,
                    "symbol": st.symbol,
                    "sell_price": sell_price,
                    "buy_price": buy_price,
                    "side": st.side,
                }
                result_queue.put(pl_data)

        except Exception as e:
            self.logger.critical(
                self.msg.get(
                    "errors.polling_thread_error",
                    thread_name=threading.current_thread().name,
                    reason=e,
                ),
                exc_info=True,
            )
            error_info = {"symbol": st.symbol, "exception": e}
            error_queue.put(error_info)

        finally:
            st.dm.close()

    # Ctrl+C ハンドラー
    def signal_handler(self, sig, frame):
        self.logger.warning(self.msg.get("info.terminate"))
        self.stop_event.set()  # スレッド停止イベントを設定
        self.exit_code = 1

    def display_profitloss(self):
        # 損益を表示する
        pl_sum = 0
        list_result = []
        today = date.today().strftime("%Y-%m-%d")

        self.logger.info("--- 損益計算結果 ---")
        for pl in self.profit_loss.values():
            if pl[2] is not None and pl[3] is not None:
                diff = pl[2] - pl[3]
                diff_str = f"{diff:,.0f}"
                sp = f"{pl[2]:,.0f}"
                bp = f"{pl[3]:,.0f}"
                self.logger.info(
                    f"[yellow]{pl[0]}[/] ({pl[1]}): 売値 = {sp} 円, 買値 = {bp} 円: 損益 = {diff_str} 円"
                )
                pl_sum += diff
                list_result.append([today, pl[0], pl[1], sp, bp, diff_str, pl[4]])
            else:
                self.logger.warning(
                    f"{pl[0]} ({pl[1]}): 売値・買値を特定できませんでした。"
                )
        pl_sum = f"{pl_sum:,.0f}"
        self.logger.info("--------------------")
        self.logger.info(f"[red]合計損益[/]: {pl_sum} 円")

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

        result = pd.DataFrame(
            [[date.today().strftime("%Y-%m-%d"), self.wallet_cash, pl_sum]],
            columns=["date", "cash", "profit_loss"],
        )
        self.dm.save_result(result)

    def main(self):
        threads = []

        try:
            self.setup_environment()

            # メインループ
            threads = self.run_main_loop()

        except (ConfigurationError, APIError) as e:
            self.logger.critical(e)
            self.exit_code = 1

        except DataProcessingError as e:
            self.logger.critical(e)
            self.exit_code = 1

        except RuntimeError as e:
            self.logger.critical(self.msg.get("errors.thread_launch_failed", reason=e))
            self.exit_code = 1

        except Exception as e:
            self.logger.critical(
                self.msg.get("errors.thread_unexpected_error", reason=e), exc_info=True
            )
            self.exit_code = 1

        finally:
            self.stop_event.set()
            self.dm.close()

            # すべてのスレッドが終了するのを待つ
            for thread in threads:
                thread.join()

            # スレッドからの最終損益報告をチェック
            self.process_final_pl_queue()

            # 結果表示
            if self.exit_code == 0:
                self.process_profitloss()

            self.logger.info(self.msg.get("info.program_end"))
            sys.exit(self.exit_code)


if __name__ == "__main__":
    # 土日祝日は実行しない
    if Misc.check_day_type(date.today()):
        print("本日は土日祝日です。プログラムを終了します。")
        sys.exit(0)

    stocktrading = StockTrading()
    stocktrading.main()
