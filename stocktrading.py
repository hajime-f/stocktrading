import time
import random
import schedule
import threading

from library import StockLibrary
from model import ModelLibrary
from stock import Stock

        
if __name__ == '__main__':

    # 株ライブラリを初期化する
    lib = StockLibrary()

    # 登録銘柄リストからすべての銘柄を削除する
    lib.unregister_all()

    # TOPIX100
    symbols = [1925, 1928, 2413, 2502, 2503, 2802, 2914, 3382, 3402, 3407, 4063, 4188, 4452, 4502, 4503, 4507, 4519, 4523, 4528, 4543, 4568, 4578, 4661, 4689, 4901, 4911, 5020, 5108, 5401, 5713, 5802, 6098, 6178, 6273, 6301, 6326, 6367, 6501, 6503, 6586, 6594, 6645, 6702, 6752, 6758, 6861, 6869, 6902, 6920, 6954, 6971, 6981, 7011, 7201, 7203, 7267, 7269, 7270, 7309, 7733, 7741, 7751, 7832, 7974, 8001, 8002, 8031, 8035, 8053, 8058, 8113, 8267, 8306, 8308, 8309, 8316, 8411, 8591, 8601, 8604, 8630, 8697, 8725, 8750, 8766, 8801, 8802, 8830, 9020, 9021, 9022, 9202, 9432, 9433, 9434, 9735, 9843, 9983, 9984,]
    # symbols = [1475,]  # テスト用銘柄

    # TOPIX100から無作為に50種類の銘柄を抽出する
    # （50銘柄に限定するのはkabuステーションの仕様に基づく）
    n_symbols = len(symbols)
    if n_symbols > 50:
        symbols = random.sample(symbols, 50)
        n_symbols = 50

    # 銘柄登録
    for s in symbols:
        lib.register(s)
        time.sleep(0.2)

    # モデルライブラリを初期化する
    model = ModelLibrary(n_symbols)
    model.load_model('./model_20250210_165310.pkl')

    # 預金残高（現物の買付余力）を問い合わせる
    deposit_before = lib.deposit()
    print(f"\033[33m買付余力：{int(deposit_before):,} 円\033[0m")
    
    # Stockクラスをインスタンス化してリストに入れる
    stocks = []
    for s in symbols:
        st = Stock(s, lib, model)
        st.set_infomation()  # 銘柄情報の設定
        stocks.append(st)
    
    # PUSH配信を受信した時に呼ばれる関数
    def receive(data):

        # 受信したデータに対応する銘柄のインスタンスを取得する
        received_stock = next(filter(lambda st: st.symbol == int(data['Symbol']), stocks), None)
        
        if received_stock:
            print(f"{data['CurrentPriceTime']}: {data['Symbol']} {received_stock.disp_name} {data['CurrentPrice']} {data['TradingVolume']}")
            received_stock.append_data(data)
        else:
            print(f"{data['Symbol']}：受信したデータに対応する銘柄が見つかりません。")
        
    # 受信関数を登録
    lib.register_receiver(receive)

    # スケジューラの定義
    def run_scheduler():
        
        while True:
            schedule.run_pending()
            time.sleep(1)

    # １分間隔でStockクラスのpolling関数を呼ぶように設定する
    for st in stocks:
        schedule.every(1).minutes.do(lambda: st.polling())

    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    try:
        lib.run()
    except KeyboardInterrupt:
        pass

    deposit_after = lib.deposit()
    print(f"\033[33m買付余力：{int(deposit_after):,} 円\033[0m")
    print(f"利益：{deposit_before - deposit_after} 円")
    
    filename = model.save_data()
    print(f"{filename}にデータを保存しました。")
        
    
