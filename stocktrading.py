import random
import signal
import sys
import threading
import time
from datetime import datetime, date
from typing import Dict

import pandas as pd

from data_manager import DataManager
from library import StockLibrary
from logger import Logger
from misc import Misc
from stock import Stock

# 定数の定義
POLLING_INTERVAL = 300  # ポーリング間隔 (秒)
POLLING_INTERVAL_VARIATION = 30  # ポーリング間隔の変動幅 (秒)

# ロガーを初期化
logger = Logger()

# スレッドを停止させるためのイベント
stop_event = threading.Event()

# 銘柄データを保持する辞書
stocks: Dict[str, Stock] = {}

# 損益を計算する辞書
profit_loss: Dict[str, list] = {}


# PUSH配信を受信した時に呼ばれる関数
def receive(data: Dict):
    # 受信したデータに対応する銘柄を取得する
    symbol = data.get("Symbol")

    # データを追加する
    if symbol in stocks:
        stocks[symbol].append_data(data)


# 約５分間隔でstockクラスのpolling関数を呼ぶように設定する
def run_polling(st):
    logger.info(f"[blue]{st.disp_name} ({st.symbol}): 取引を開始します。[/]")

    while not stop_event.is_set():
        time.sleep(random.uniform(0, POLLING_INTERVAL_VARIATION))
        st.polling()
        time.sleep(POLLING_INTERVAL)

    # while文を抜けたときに実行する処理
    if st.check_transaction():
        sell_price, buy_price = st.fetch_prices()
        profit_loss[st.symbol] = [
            st.disp_name,
            st.symbol,
            sell_price,
            buy_price,
            st.side,
        ]


# Ctrl+C ハンドラー
def signal_handler(sig, frame):
    logger.warning("[red]Ctrl+C が押されました。終了処理を行います。[/]")
    stop_event.set()  # スレッド停止イベントを設定


def display_profitloss():
    # 損益を表示する
    pl_sum = 0
    list_result = []
    today = date.today().strftime("%Y-%m-%d")

    logger.info("--- 損益計算結果 ---")
    for pl in profit_loss.values():
        if pl[2] is not None and pl[3] is not None:
            diff = pl[2] - pl[3]
            logger.info(
                f"{pl[0]} ({pl[1]}): 売値 = {pl[2]:,.0f} 円, 買値 = {pl[3]:,.0f} 円: 損益 = {diff:,.0f} 円"
            )
            pl_sum += diff
            list_result.append([today, pl[0], pl[1], pl[2], pl[3], diff, pl[4]])
        else:
            logger.warning(f"{pl[0]} ({pl[1]}): 売値・買値を特定できませんでした。")
    logger.info("--------------------")
    logger.info(f"合計損益: {pl_sum:,.0f} 円")

    return pl_sum, list_result


if __name__ == "__main__":
    # 土日祝日は実行しない
    if Misc().check_day_type(date.today()):
        exit()

    # 株ライブラリを初期化
    lib = StockLibrary(logger)

    # 登録銘柄リストからすべての銘柄を削除する
    lib.unregister_all()

    # 今回取引する銘柄リストを取得
    dm = DataManager()
    # target_stocks = dm.fetch_target(table_name="Target", target_date="2025-05-15")
    target_symbols = [
        ["2025-06-02", "1475", "iシェアーズ・コア TOPIX ETF", 0.999, 1],
        ["2025-06-02", "1592", "上場インデックス JPX日経インデックス400", 0.999, 2],
        ["2025-06-02", "1586", "上場インデックス TOPIX Ex-Financials", 0.999, 1],
        ["2025-06-02", "1481", "上場インデックスファンド日本経済貢献株", 0.999, 2],
        ["2025-06-02", "1578", "上場インデックスファンド日経225(ミニ)", 0.999, 1],
        ["2025-06-02", "2552", "上場Ｊリート(東証REIT指数)隔月分配(ミニ)", 0.999, 2],
    ]
    columns = ["date", "code", "brand", "pred", "side"]
    target_stocks = pd.DataFrame(target_symbols, columns=columns)

    # 銘柄登録
    lib.register(target_stocks["code"].tolist())

    # 取引余力を取得
    wallet_margin = lib.wallet_margin()
    logger.info(f"[yellow]取引余力（信用）：{int(wallet_margin):,} 円[/]")
    wallet_cash = lib.wallet_cash()
    logger.info(f"[yellow]取引余力（現物）：{int(wallet_cash):,} 円[/]")

    # Stockクラスをインスタンス化して辞書に入れる
    for _, row in target_stocks.iterrows():
        symbol = row["code"]
        stock_instance = Stock(symbol, lib, dm, logger, row["side"], row["brand"])
        stock_instance.set_information()
        stocks[symbol] = stock_instance

    # 受信関数を登録
    lib.register_receiver(receive)

    # Ctrl+C ハンドラーを登録
    signal.signal(signal.SIGINT, signal_handler)

    # スレッドを準備
    threads = [
        threading.Thread(target=run_polling, args=(st,)) for st in stocks.values()
    ]
    push_receiver_thread = threading.Thread(target=lib.run, daemon=True)

    # スレッドを起動
    try:
        logger.info("スレッドを起動しています。")
        for thread in threads:
            thread.start()
            push_receiver_thread.start()
        logger.info(
            "[green]すべてのスレッドを起動しました。プログラムは 15:35 に自動終了します。[/]"
        )

        while True:
            now = datetime.now()
            if now.hour > 15 or (now.hour == 15 and now.minute >= 35):
                logger.info("[green]終了処理を開始します。[/]")
                stop_event.set()

            # 停止イベントがセットされたら、監視ループを抜ける（Ctrl+Cまたは時間経過）
            if stop_event.is_set():
                break

            # 10秒ごとにチェック
            time.sleep(10)

    except RuntimeError:
        logger.error("[bold red]スレッドの起動に失敗しました。[/]", exc_info=True)
        stop_event.set()
        sys.exit(1)

    except Exception:
        logger.error(
            "[bold red]メインスレッドで予期せぬエラーが発生しました。[/]",
            exc_info=True,
        )
        stop_event.set()
        sys.exit(1)

    finally:
        # すべてのスレッドが終了するのを待つ
        for thread in threads:
            thread.join()

        # 損益を表示する
        pl_sum, list_result = display_profitloss()

        # 損益を記録
        plofit_loss = pd.DataFrame(
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
        dm.save_profit_loss(profit_loss)

        result = pd.DataFrame(
            [date.today().strftime("%Y-%m-%d"), wallet_cash, pl_sum],
            columns=["date", "cash", "profit_loss"],
        )
        dm.save_result(result)

        sys.exit(0)
