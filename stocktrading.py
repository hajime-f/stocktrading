import random
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

from rich.console import Console

from data_manager import DataManager
from library import StockLibrary
from stock import Stock

# 定数の定義
POLLING_INTERVAL: int = 300  # ポーリング間隔 (秒)
POLLING_INTERVAL_VARIATION: int = 30  # ポーリング間隔の変動幅 (秒)

console = Console(log_time_format="%Y-%m-%d %H:%M:%S")

# スレッドを停止させるためのイベント
stop_event = threading.Event()

# 銘柄データを保持する辞書
stocks: Dict[str, Stock] = {}


# PUSH配信を受信した時に呼ばれる関数
def receive(data: Dict):
    """
    PUSH配信を受信した時に呼ばれる関数
    """

    # 受信したデータに対応する銘柄のインスタンスを取得する
    symbol = data.get("Symbol")

    # データを追加する
    if symbol in stocks:
        stocks[symbol].append_data(data)


# 約５分間隔でstockクラスのpolling関数を呼ぶように設定する
def run_polling(st: Stock):
    """
    約５分間隔でstockクラスのpolling関数を呼ぶ関数
    """

    while not stop_event.is_set():
        try:
            stop_event.wait(random.uniform(0, POLLING_INTERVAL_VARIATION))
            st.polling()
            stop_event.wait(POLLING_INTERVAL)
        except Exception as e:
            console.log(f"[red]エラーが発生しました: {e}[/]")
            time.sleep(5)


# Ctrl+C ハンドラー
def signal_handler(sig, frame):
    """
    Ctrl+C ハンドラー
    """

    console.log("[red]Ctrl+C が押されました。終了処理を行います。[/]")
    if not stop_event.is_set():
        stop_event.set()  # スレッド停止イベントを設定


if __name__ == "__main__":
    # 株ライブラリを初期化
    lib = StockLibrary()

    # 登録銘柄リストからすべての銘柄を削除する
    lib.unregister_all()

    # 今回取引する銘柄リストを取得
    dm = DataManager()
    # target_symbols = [symbol[1] for symbol in dm.fetch_target()]
    target_symbols = ["1475", "1592", "1586", "1481", "1578", "2552"]  # テスト用銘柄

    # 銘柄登録
    lib.register(target_symbols)

    # 取引余力を取得
    wallet_margin = lib.wallet_margin()
    console.log(f"[yellow]取引余力（信用）：{int(wallet_margin):,} 円[/]")
    wallet_cash = lib.wallet_cash()
    console.log(f"[yellow]取引余力（現物）：{int(wallet_cash):,} 円[/]")

    stocks = {}
    for symbol in target_symbols:
        stock_instance = Stock(symbol, lib, dm)
        if stock_instance.set_information():
            stocks[symbol] = stock_instance

    # 受信関数を登録
    lib.register_receiver(receive)

    # Ctrl+C ハンドラーを登録
    signal.signal(signal.SIGINT, signal_handler)

    # スレッド起動
    executor = ThreadPoolExecutor(
        max_workers=len(stocks),
        thread_name_prefix="PollingThread",
    )
    polling_futures = [executor.submit(run_polling, st) for st in stocks.values()]

    try:
        lib.run()
    except Exception as e:
        console.log(f"[red]エラーが発生しました: {e}[/]")
        if not stop_event.is_set():
            stop_event.set()
    finally:
        if not stop_event.is_set():
            stop_event.set()

        # すべてのスレッドが終了するのを待つ
        executor.shutdown(wait=True)

        # 全スレッド終了後に各銘柄の終了時処理を実行
        for symbol, stock_instance in stocks.items():
            stock_instance.check_transaction()

        # 損益を計算する
        pl = dm.calc_profitloss()
        console.log("--- 損益計算結果 ---")
        console.print(pl)
        console.log("--------------------")
        console.log(f"合計損益: {pl['Result'].sum():,.0f} 円")
        console.log("--------------------")
