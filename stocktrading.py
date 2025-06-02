import random
import signal
import sys
import threading
import time
from typing import Dict

import pandas as pd
from rich.console import Console

from data_manager import DataManager
from library import StockLibrary
from stock import Stock

# 定数の定義
POLLING_INTERVAL = 300  # ポーリング間隔 (秒)
POLLING_INTERVAL_VARIATION = 15  # ポーリング間隔の変動幅 (秒)

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
def run_polling(st):
    """
    約５分間隔でstockクラスのpolling関数を呼ぶ関数
    """

    while not stop_event.is_set():
        time.sleep(random.uniform(0, POLLING_INTERVAL_VARIATION))
        st.polling()
        time.sleep(POLLING_INTERVAL)

    # Ctrl+C が押されたときに実行する処理
    st.check_transaction()


# Ctrl+C ハンドラー
def signal_handler(sig, frame):
    """
    Ctrl+C ハンドラー
    """

    console.log("[red]Ctrl+C が押されました。終了処理を行います。[/]")
    stop_event.set()  # スレッド停止イベントを設定
    sys.exit(0)  # プログラムを終了


if __name__ == "__main__":
    # 株ライブラリを初期化
    lib = StockLibrary()

    # 登録銘柄リストからすべての銘柄を削除する
    lib.unregister_all()

    # 今回取引する銘柄リストを取得
    dm = DataManager()
    # target_stocks = dm.fetch_target(table_name="Target", target_date="2025-05-15")
    target_symbols = [
        ["2025-06-02", "1475", "iシェアーズ・コア TOPIX ETF", 0.999, 1],
        ["2025-06-02", "1592", "上場インデックス JPX日経インデックス400", 0.999, 2],
        # ["2025-06-02", "1586", "上場インデックス TOPIX Ex-Financials", 0.999, 1],
        # ["2025-06-02", "1481", "上場インデックスファンド日本経済貢献株", 0.999, 2],
        # ["2025-06-02", "1578", "上場インデックスファンド日経225(ミニ)", 0.999, 1],
        # ["2025-06-02", "2552", "上場Ｊリート(東証REIT指数)隔月分配(ミニ)", 0.999, 2],
    ]
    columns = ["date", "code", "brand", "pred", "side"]
    target_stocks = pd.DataFrame(target_symbols, columns=columns)

    # 銘柄登録
    lib.register(target_stocks["code"].tolist())

    # 取引余力を取得
    wallet_margin = lib.wallet_margin()
    console.log(f"[yellow]取引余力（信用）：{int(wallet_margin):,} 円[/]")
    wallet_cash = lib.wallet_cash()
    console.log(f"[yellow]取引余力（現物）：{int(wallet_cash):,} 円[/]")

    # Stockクラスをインスタンス化して辞書に入れる
    for _, row in target_stocks.iterrows():
        symbol = row["code"]
        stock_instance = Stock(symbol, lib, dm, row["side"], row["brand"])
        stock_instance.set_information()
        stocks[symbol] = stock_instance

    # 受信関数を登録
    lib.register_receiver(receive)

    # Ctrl+C ハンドラーを登録
    signal.signal(signal.SIGINT, signal_handler)

    # スレッド起動
    threads = [
        threading.Thread(target=run_polling, args=(st,)) for st in stocks.values()
    ]
    for thread in threads:
        thread.start()

    try:
        lib.run()
    except Exception as e:
        console.log(f"[red]エラーが発生しました: {e}[/]")
    finally:
        # すべてのスレッドが終了するのを待つ
        for thread in threads:
            thread.join()

        # 損益を計算する
        pl = dm.calc_profitloss()
        console.log("--- 損益計算結果 ---")
        console.log(pl)
        console.log(f"合計損益: {pl['Result'].sum():,.0f} 円")
        console.log("--------------------")
