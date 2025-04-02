import datetime
import random
import signal
import sys
import threading

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
    global stocks
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
        wait_time = (
            POLLING_INTERVAL + (2 * random.random() - 1) * POLLING_INTERVAL_VARIATION
        )
        if stop_event.wait(timeout=wait_time):
            break

        if not stop_event.is_set():
            st.polling()

    # Ctrl+C または時刻超過でループが停止した後に実行する処理
    st.check_transaction()


# 時刻監視スレッド用の関数
def time_watcher():
    """
    時刻を監視し、15:30を過ぎたら停止イベントを設定する関数
    """

    target_time = datetime.time(15, 30)

    while not stop_event.is_set():
        now = datetime.datetime.now()
        current_time = now.time()

        # 目標時刻を過ぎたら停止処理
        if current_time >= target_time:
            stop_event.set()
            sys.exit(0)

        # 10秒待つか、イベントがセットされたら抜ける
        if stop_event.wait(timeout=10):
            break


# Ctrl+C ハンドラー
def signal_handler(sig, frame):
    """
    Ctrl+C ハンドラー
    """

    # 既に停止処理が始まっている場合は何もしない
    if stop_event.is_set():
        return

    # スレッド停止イベントを設定
    console.log("[red]Ctrl+C が押されました。終了処理を行います。[/]")
    stop_event.set()  # スレッド停止イベントを設定
    sys.exit(0)  # プログラムを終了


if __name__ == "__main__":
    # 取引のベース単位
    # このシステムでは、ベース単位（base_transaction）×単元株（stock.unit）だけ取引を実行する。
    # 例えば、ベース単位を５に設定すると、単元株が100株の銘柄であれば、毎回500株取引することになる。
    # 当然、ベース単位を引き上げるほど取引価格が上がっていくので、注意が必要。
    base_transaction = 1

    stocks = []
    dm, lib = None, None

    try:
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

        # Stockクラスをインスタンス化してリストに入れる (グローバル変数 stocks に代入)
        stocks = [Stock(s, lib, dm, base_transaction) for s in symbols]
        for st in stocks:
            st.set_information()  # 銘柄情報の設定

        # 受信関数を登録
        lib.register_receiver(receive)

        # Ctrl+C ハンドラーを登録
        signal.signal(signal.SIGINT, signal_handler)

        # ポーリングスレッド起動
        threads = []
        for st in stocks:
            thread = threading.Thread(
                target=run_polling, args=(st,), name=f"Polling-{st.symbol}"
            )
            threads.append(thread)
            thread.start()

        # 時刻監視スレッドを起動
        time_thread = threading.Thread(
            target=time_watcher, daemon=True, name="TimeWatcher"
        )
        time_thread.start()

        # メイン処理
        lib.run()

        # lib.run() が何らかの理由で正常終了した場合の処理
        if not stop_event.is_set():
            stop_event.set()

    except SystemExit:  # sys.exit(0) を捕捉
        console.log("プログラム終了シグナル (SystemExit) を受け取りました。")

    except KeyboardInterrupt:  # 稀に signal_handler 前に補足される場合
        console.log("[red]KeyboardInterrupt を検知しました。終了処理を開始します。[/]")
        if not stop_event.is_set():
            stop_event.set()

    except Exception as e:
        console.log(f"[red]予期せぬエラーが発生しました: {e}[/]")
        import traceback

        traceback.print_exc()  # 詳細なエラー情報を表示
        if not stop_event.is_set():
            stop_event.set()  # エラー発生時も他のスレッドを止める

    finally:
        # すべてのポーリングスレッドが終了するのを待つ
        for thread in threads:
            if thread.is_alive():
                thread.join(timeout=POLLING_INTERVAL)
                if thread.is_alive():
                    console.log(
                        f"[yellow]警告: スレッド {thread.name} が時間内に終了しませんでした。[/yellow]"
                    )

        # 損益を計算する
        if dm is not None:
            try:
                pl = dm.calc_profitloss()
                console.log("--- 損益計算結果 ---")
                console.log(pl)
                console.log(f"合計損益: {pl['Result'].sum():,.0f} 円")
                console.log("--------------------")
            except Exception as e:
                console.log(f"[red]損益計算中にエラーが発生しました: {e}[/]")
        else:
            console.log(
                "[yellow]DataManagerが初期化されていないため、損益計算をスキップします。[/yellow]"
            )

        # StockLibrary のクリーンアップ処理などがあればここに追加
        if lib is not None:
            try:
                # lib.cleanup() # 例: もしクリーンアップメソッドがあれば呼び出す
                pass
            except Exception as e:
                console.log(f"[red]StockLibrary のクリーンアップ中にエラー: {e}[/]")
