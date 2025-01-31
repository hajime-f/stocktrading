from tqdm import tqdm

import stocklib
from Stock import Stock

if __name__ == '__main__':

    # ライブラリを初期化する
    lib = stocklib.Initialize()

    # 預金残高（現物の買付余力）を問い合わせる
    deposit = lib.information.deposit()
    print(f"\033[33m買付余力：{int(deposit):,} 円\033[0m")

    # 登録銘柄リストからすべての銘柄を削除する
    lib.register.unregister_all()

    # codes = [1925, 1928, 2413, 2502, 2503, 2802, 2914, 3382, 3402, 3407, 4063, 4188, 4452, 4502, 4503, 4507, 4519, 4523, 4528, 4543, 4568, 4578, 4661, 4689, 4901, 4911, 5020, 5108, 5401, 5713, 5802, 6098, 6178, 6273, 6301, 6326, 6367, 6501, 6502, 6503, 6586, 6594, 6645, 6702, 6752, 6758, 6861, 6869, 6902, 6920, 6954, 6971, 6981, 7011, 7201, 7203, 7267, 7269, 7270, 7309, 7733, 7741, 7751, 7832, 7974, 8001, 8002, 8031, 8035, 8053, 8058, 8113, 8267, 8306, 8308, 8309, 8316, 8411, 8591, 8601, 8604, 8630, 8697, 8725, 8750, 8766, 8801, 8802, 8830, 9020, 9021, 9022, 9202, 9432, 9433, 9434, 9735, 9843, 9983, 9984,]  # TOPIX100
    codes = [1475,]  # テスト用銘柄

    # 登録できる銘柄の上限数は50のため数を絞る
    codes = codes[0:50]

    # bar = tqdm(total = len(codes))
    # bar.set_description('ポートフォリオを初期化しています')
    
    # Stockクラスをインスタンス化してリストに入れる
    stocks = []
    for c in codes:
        st = Stock(c, lib)
        st.register_to_list()  # 銘柄登録
        st.set_infomation()  # 銘柄情報の設定
        stocks.append(st)
        # bar.update(1)
    

    @lib.websocket
    def receive(msg):
        print(msg)

    lib.websocket.run()    
    
    
