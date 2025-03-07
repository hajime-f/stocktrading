import random

from library import StockLibrary
from data_management import DataManagement


if __name__ == '__main__':

    # 株ライブラリを初期化する
    lib = StockLibrary()

    # 登録銘柄リストからすべての銘柄を削除する
    lib.unregister_all()

    # TOPIX100
    symbols = [1925, 1928, 2413, 2502, 2503, 2802, 2914, 3382, 3402, 3407, 4063, 4188, 4452, 4502, 4503, 4507, 4519, 4523, 4528, 4543, 4568, 4578, 4661, 4689, 4901, 4911, 5020, 5108, 5401, 5713, 5802, 6098, 6178, 6273, 6301, 6326, 6367, 6501, 6503, 6586, 6594, 6645, 6702, 6752, 6758, 6861, 6869, 6902, 6920, 6954, 6971, 6981, 7011, 7201, 7203, 7267, 7269, 7270, 7309, 7733, 7741, 7751, 7832, 7974, 8001, 8002, 8031, 8035, 8053, 8058, 8113, 8267, 8306, 8308, 8309, 8316, 8411, 8591, 8601, 8604, 8630, 8697, 8725, 8750, 8766, 8801, 8802, 8830, 9020, 9021, 9022, 9202, 9432, 9433, 9434, 9735, 9843, 9983, 9984,]

    # TOPIX100から無作為に50種類の銘柄を抽出する
    # （50銘柄に限定するのはkabuステーションの仕様に基づく）
    n_symbols = len(symbols)
    if n_symbols > 50:
        symbols = random.sample(symbols, 50)
        n_symbols = 50

    # 銘柄登録
    lib.register(symbols)

    # データライブラリを初期化する
    dm = DataManagement(n_symbols)

    def receive(data):

        # 受信したデータに対応する銘柄のインデクスを取得する
        try:
            index = symbols.index(int(data['Symbol']))
        except ValueError:
            print("受信したデータに対応する銘柄が見つかりません。")
            exit()

        # 情報表示
        print(f"{data['CurrentPriceTime']}: {data['Symbol']} {data['SymbolName']} {data['CurrentPrice']} {data['TradingVolume']}")

        # データを追加する
        dm.append_data(data, index)

    # 受信関数を登録
    lib.register_receiver(receive)

    try:
        lib.run()
    except KeyboardInterrupt:
        pass
    
    # データを整理する
    print("データを整理しています。")
    data_list = dm.prepare_dataframe_list(symbols)

    # データを保存する
    filename = dm.save_data(data_list)
    print(f"{filename} にデータを保存しました。")
    
        
