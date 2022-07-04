from Stock import Stock
from tqdm import tqdm
from dataclasses import field, dataclass
import time

@dataclass(slots = True)
class Portfolio:

    deposit: int = 0    
    stocks: list[int] = field(default_factory = list)
    

def init_portfolio(key):

    # 預金残高（現物の買付余力）をサーバに問い合わせる
    deposit = key.inquiry_deposit()
    print(f"\033[33m預金残高：{int(deposit):,} 円\033[0m")

    codes = [1925, 1928, 2413, 2502, 2503, 2802, 2914, 3382, 3402, 3407, 4063, 4188, 4452, 4502, 4503, 4507, 4519, 4523, 4528, 4543, 4568, 4578, 4661, 4689, 4901, 4911, 5020, 5108, 5401, 5713, 5802, 6098, 6178, 6273, 6301, 6326, 6367, 6501, 6502, 6503, 6586, 6594, 6645, 6702, 6752, 6758, 6861, 6869, 6902, 6920, 6954, 6971, 6981, 7011, 7201, 7203, 7267, 7269, 7270, 7309, 7733, 7741, 7751, 7832, 7974, 8001, 8002, 8031, 8035, 8053, 8058, 8113, 8267, 8306, 8308, 8309, 8316, 8411, 8591, 8601, 8604, 8630, 8697, 8725, 8750, 8766, 8801, 8802, 8830, 9020, 9021, 9022, 9202, 9432, 9433, 9434, 9735, 9843, 9983, 9984,]  # TOPIX100
    # codes = [1475,]  # テスト用銘柄

    # 登録できる銘柄の上限数は50のため数を絞る
    codes = codes[0:50]
    
    bar = tqdm(total = len(codes))
    bar.set_description('ポートフォリオを初期化しています')
    
    # Stock クラスをインスタンス化してリストに入れる
    stocks = []
    for c in codes:
        st = Stock(c, key)
        st.register_to_stock_list()  # 銘柄登録
        st.set_brand_data()  # 銘柄情報の設定
        time.sleep(0.2)  # 流量制限を回避するためのスリープ
        stocks.append(st)
        bar.update(1)

    # ポートフォリオを構成する
    pf = Portfolio(deposit, stocks)
    
    return pf
    
    
def collect_information(stock):
    
    # 最新のデータを取得する
    stock.fetch_data()
    print('データ取得：', stock.code, stock.disp_name)
    time.sleep(0.3)
    
    return 1


def execute_transaction(stock):
    
    return 1


def trade(pf):
    
    for st in pf.stocks:
        
        info = collect_information(st)
        
        trans = execute_transaction(st)
        
    
    return True


def finalize(pf):
    
    for st in pf.stocks:
        st.store_daily_candle()  # 日足データの保存
        st.unregister_from_stock_list()  # 銘柄登録の解除
        time.sleep(0.2)  # 流量制限を回避するためのスリープ

