import random
import signal
import sys
import threading
import time

from rich.console import Console

from data_manager import DataManager
from library import StockLibrary
from stock import Stock

console = Console(log_time_format="%Y-%m-%d %H:%M:%S")

# スレッドを停止させるためのフラグ
stop_threads = False

if __name__ == "__main__":
    # 取引のベース単位
    # このシステムでは、ベース単位（base_transaction）×単元株（stock.unit）だけ取引を実行する。
    # 例えば、ベース単位を５に設定すると、単元株が100株の銘柄であれば、毎回500株取引することになる。
    # 当然、ベース単位を引き上げるほど取引価格が上がっていくので、注意が必要。
    base_transaction = 1

    # 株ライブラリを初期化する
    lib = StockLibrary()

    # 登録銘柄リストからすべての銘柄を削除する
    lib.unregister_all()

    # 今回取引する銘柄のリストを取得する
    dm = DataManager()
    # symbols = [symbol[1] for symbol in dm.fetch_target()]
    # symbols = ["1329", "1475", "1592", "1586", "1481", "1578", "2552"]  # テスト用銘柄
    symbols = ["1475"]

    # 銘柄登録
    lib.register(symbols)

    # 預金残高（現物の買付余力）を問い合わせる
    deposit_before = lib.deposit()
    console.log(f"[yellow]買付余力：{int(deposit_before):,} 円[/]")

    # Stockクラスをインスタンス化してリストに入れる
    stocks = []
    for s in symbols:
        st = Stock(s, lib, dm, base_transaction)
        st.set_information()  # 銘柄情報の設定
        stocks.append(st)

    # PUSH配信を受信した時に呼ばれる関数
    def receive(data):
        # 受信したデータに対応する銘柄のインスタンスを取得する
        received_stock = next(
            filter(lambda st: st.symbol == data["Symbol"], stocks),
            None,
        )

        # データを追加する
        if received_stock:
            received_stock.append_data(data)

    # 受信関数を登録
    lib.register_receiver(receive)

    # 約５分間隔でstockクラスのpolling関数を呼ぶように設定する
    def run_polling(st):
        while True:
            time.sleep(300 + (2 * random.random() - 1) * 30)
            st.polling()

    # Ctrl+C ハンドラー
    def signal_handler(sig, frame):
        global stop_threads
        console.log("[red]Ctrl+C が押されました。終了処理を行います。[/]")
        stop_threads = True  # スレッド停止フラグを設定
        sys.exit(0)  # プログラムを終了

    # Ctrl+C ハンドラーを登録
    signal.signal(signal.SIGINT, signal_handler)

    # スレッド起動
    threads = []
    for st in stocks:
        thread = threading.Thread(target=run_polling, args=(st,))
        threads.append(thread)
        thread.start()

    try:
        lib.run()
    except Exception as e:
        console.log(f"[red]エラーが発生しました: {e}")
    finally:
        # すべてのスレッドが終了するのを待つ
        for thread in threads:
            thread.join()

        deposit_after = lib.deposit()
        console.log(f"[yellow]買付余力：{int(deposit_after):,} 円[/]")
        console.log(f"[yellow]損益：{int(deposit_before - deposit_after):,} 円[/]")

        df = dm.load_order()
        price_side_1 = (
            df[df["Side"] == 1]["Price"].iloc[0]
            if not df[df["Side"] == 1].empty
            else None
        )
        price_side_2 = (
            df[df["Side"] == 2]["Price"].iloc[0]
            if not df[df["Side"] == 2].empty
            else None
        )
        if price_side_1 is not None and price_side_2 is not None:
            tmp = price_side_1 - price_side_2
        breakpoint()
