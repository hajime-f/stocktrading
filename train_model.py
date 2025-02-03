import time
from datetime import datetime

import stocklib
import model

if __name__ == '__main__':

    # ライブラリを初期化する
    lib = stocklib.StockLibrary()

    # 登録銘柄リストからすべての銘柄を削除する
    lib.register.unregister_all()

    symbols = [1925, 1928, 2413, 2502, 2503, 2802, 2914, 3382, 3402, 3407, 4063, 4188, 4452, 4502, 4503, 4507, 4519, 4523, 4528, 4543, 4568, 4578, 4661, 4689, 4901, 4911, 5020, 5108, 5401, 5713, 5802, 6098, 6178, 6273, 6301, 6326, 6367, 6501, 6503, 6586, 6594, 6645, 6702, 6752, 6758, 6861, 6869, 6902, 6920, 6954, 6971, 6981, 7011, 7201, 7203, 7267, 7269, 7270, 7309, 7733, 7741, 7751, 7832, 7974, 8001, 8002, 8031, 8035, 8053, 8058, 8113, 8267, 8306, 8308, 8309, 8316, 8411, 8591, 8601, 8604, 8630, 8697, 8725, 8750, 8766, 8801, 8802, 8830, 9020, 9021, 9022, 9202, 9432, 9433, 9434, 9735, 9843, 9983, 9984,]  # TOPIX100
    
    # 登録できる銘柄の上限数は50のため数を絞る
    symbols = symbols[0:50]

    # モデルライブラリを初期化する
    n_symbols = len(symbols)
    model = model.Model(n_symbols)
    lib.websocket.set_model(model)

    # 銘柄登録
    for s in symbols:
        lib.register.register(s)
        time.sleep(0.3)
        
    @lib.websocket
    def receive(data):

        # 受信したデータに対応する銘柄のインデクスを取得する
        try:
            index = symbols.index(int(data['Symbol']))
        except ValueError:
            print("受信したデータに対応する銘柄が見つかりません。")
            exit()
            
        # データを受信した時刻を取得する
        try:
            dt_object = datetime.strptime(data['CurrentPriceTime'], "%Y-%m-%dT%H:%M:%S%z")  
            formatted_datetime = dt_object.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            print("文字列のフォーマットが異なります。")
            exit()

        print(f"{data['CurrentPriceTime']}: {data['Symbol']} {data['SymbolName']} {data['CurrentPrice']} {data['TradingVolume']}")

        # データを追加する
        model.train.append_data(formatted_datetime, data['CurrentPrice'], index)

    try:
        lib.websocket.run()
    except KeyboardInterrupt:
        
        model.train.train_model()
        
        
