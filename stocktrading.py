import os
import time
import threading

from library import StockLibrary
from model import ModelLibrary
from stock import Stock
from data_management import DataManagement


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

    # 銘柄リストを取得する
    dm = DataManagement()
    # symbols = [symbol[1] for symbol in dm.fetch_target()]
    # symbols = ['1329', '1475', '1592', '1586', '1481', '1578', '2552',]  # テスト用銘柄
    symbols = [
        "5481",
    ]

    # 銘柄登録
    lib.register(symbols)

    # モデルライブラリを初期化する
    model = ModelLibrary()
    filename = os.path.join("./model/", "model_daytrade_20250313_085957.keras")
    model.load_model(filename)

    # 預金残高（現物の買付余力）を問い合わせる
    deposit_before = lib.deposit()
    print(f"\033[33m買付余力：{int(deposit_before):,} 円\033[0m")

    # Stockクラスをインスタンス化してリストに入れる
    stocks = []
    for s in symbols:
        st = Stock(s, lib, model, base_transaction)
        st.set_information()  # 銘柄情報の設定
        stocks.append(st)

    # PUSH配信を受信した時に呼ばれる関数
    def receive(data):
        # 受信したデータに対応する銘柄のインスタンスを取得する
        received_stock = next(
            filter(lambda st: st.symbol == data["Symbol"], stocks), None
        )

        # データを追加する
        if received_stock:
            received_stock.append_data(data)

    # 受信関数を登録
    lib.register_receiver(receive)

    # １分間隔でstockクラスのpolling関数を呼ぶように設定する
    def run_polling(st):
        while True:
            st.polling()
            time.sleep(60)

    for st in stocks:
        thread = threading.Thread(target=run_polling, args=(st,))
        thread.start()

    try:
        lib.run()
    except KeyboardInterrupt:
        pass

    deposit_after = lib.deposit()
    print(f"\033[33m買付余力：{int(deposit_after):,} 円\033[0m")
    print(f"損益：{deposit_before - deposit_after:,} 円")
