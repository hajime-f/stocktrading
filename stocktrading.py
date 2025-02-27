import os
import time
import random
import schedule
import threading

from library import StockLibrary
from model import ModelLibrary
from stock import Stock

        
if __name__ == '__main__':
    
    # 取引のベース単位
    # このシステムでは、ベース単位（base_transaction）×単元株（stock.unit）だけ取引を実行する。
    # 例えば、ベース単位を５に設定すると、単元株が100株の銘柄であれば、毎回500株取引することになる。
    # 当然、ベース単位を引き上げるほど取引価格が上がっていくので、注意が必要。
    base_transaction = 1
    
    # 株ライブラリを初期化する
    lib = StockLibrary()

    # 登録銘柄リストからすべての銘柄を削除する
    lib.unregister_all()

    # TOPIX100
    # symbols = [1925, 1928, 2413, 2502, 2503, 2802, 2914, 3382, 3402, 3407, 4063, 4188, 4452, 4502, 4503, 4507, 4519, 4523, 4528, 4543, 4568, 4578, 4661, 4689, 4901, 4911, 5020, 5108, 5401, 5713, 5802, 6098, 6178, 6273, 6301, 6326, 6367, 6501, 6503, 6586, 6594, 6645, 6702, 6752, 6758, 6861, 6869, 6902, 6920, 6954, 6971, 6981, 7011, 7201, 7203, 7267, 7269, 7270, 7309, 7733, 7741, 7751, 7832, 7974, 8001, 8002, 8031, 8035, 8053, 8058, 8113, 8267, 8306, 8308, 8309, 8316, 8411, 8591, 8601, 8604, 8630, 8697, 8725, 8750, 8766, 8801, 8802, 8830, 9020, 9021, 9022, 9202, 9432, 9433, 9434, 9735, 9843, 9983, 9984,]
    symbols = [1329, 1364, 1475, 1592, 1586, 1481, 1578, 2552,]  # テスト用銘柄

    # TOPIX100から無作為に50種類の銘柄を抽出する
    # （50銘柄に限定するのはkabuステーションの仕様に基づく）
    n_symbols = len(symbols)
    if n_symbols > 50:
        symbols = random.sample(symbols, 50)
        n_symbols = 50    
        
    # 銘柄登録
    lib.register(symbols)
    
    # モデルライブラリを初期化する
    model = ModelLibrary(n_symbols)
    filename = os.path.join("./model/", 'model_20250225_153630.pkl')
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
        received_stock = next(filter(lambda st: st.symbol == int(data['Symbol']), stocks), None)

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
        thread = threading.Thread(target = run_polling, args = (st,))
        thread.start()
    
    try:
        lib.run()
    except KeyboardInterrupt:
        pass

    deposit_after = lib.deposit()
    print(f"\033[33m買付余力：{int(deposit_after):,} 円\033[0m")
    print(f"損益：{deposit_before - deposit_after:,} 円")
    
