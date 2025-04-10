import random
import signal
import sys
import threading
import time

from rich.console import Console

from data_manager import DataManager
from library import StockLibrary
from stock import Stock

# 定数の定義
POLLING_INTERVAL = 300  # ポーリング間隔 (秒)
POLLING_INTERVAL_VARIATION = 30  # ポーリング間隔の変動幅 (秒)

console = Console(log_time_format="%Y-%m-%d %H:%M:%S")

# スレッドを停止させるためのイベント
stop_event = threading.Event()


# PUSH配信を受信した時に呼ばれる関数
def receive(data):
    """
    PUSH配信を受信した時に呼ばれる関数
    """

    # 受信したデータに対応する銘柄のインスタンスを取得する
    received_stock = next(
        filter(lambda st: st.symbol == data["Symbol"], stocks),
        None,
    )
    # データを追加する
    if received_stock:
        received_stock.append_data(data)


# 約５分間隔でstockクラスのpolling関数を呼ぶように設定する
def run_polling(st):
    """
    約５分間隔でstockクラスのpolling関数を呼ぶ関数
    """

    while not stop_event.is_set():
        time.sleep(
            POLLING_INTERVAL + (2 * random.random() - 1) * POLLING_INTERVAL_VARIATION
        )
        st.polling()

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
    # 取引のベース単位
    # このシステムでは、ベース単位（base_transaction）×単元株（stock.unit）だけ取引を実行する。
    # 例えば、ベース単位を５に設定すると、単元株が100株の銘柄であれば、毎回500株取引することになる。
    # 当然、ベース単位を引き上げるほど取引価格が上がっていくので、注意が必要。
    base_transaction = 1

    # 株ライブラリを初期化
    lib = StockLibrary()

    # 登録銘柄リストからすべての銘柄を削除する
    lib.unregister_all()

    # 今回取引する銘柄リストを取得
    dm = DataManager()
    # symbols = [symbol[1] for symbol in dm.fetch_target()]
    symbols = ["1475", "1592", "1586", "1481", "1578", "2552"]  # テスト用銘柄
    # symbols = ["1475"]  # テスト用銘柄

    # 銘柄登録
    lib.register(symbols)

    # 取引余力を取得
    wallet_margin = lib.wallet_margin()
    console.log(f"[yellow]取引余力（信用）：{int(wallet_margin):,} 円[/]")
    wallet_cash = lib.wallet_cash()
    console.log(f"[yellow]取引余力（現物）：{int(wallet_cash):,} 円[/]")

    # Stockクラスをインスタンス化してリストに入れる
    stocks = [Stock(s, lib, dm, base_transaction) for s in symbols]
    for st in stocks:
        st.set_information()  # 銘柄情報の設定

    # 受信関数を登録
    lib.register_receiver(receive)

    # Ctrl+C ハンドラーを登録
    signal.signal(signal.SIGINT, signal_handler)

    # スレッド起動
    threads = [threading.Thread(target=run_polling, args=(st,)) for st in stocks]
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
