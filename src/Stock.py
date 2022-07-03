import pandas as pd
import sqlite3, uuid, time
<<<<<<< HEAD
from datetime import datetime, timezone, timedelta
=======
from datetime import date
from datetime import datetime
>>>>>>> 625b6c6d6182a82066877ffbf19fed5b98053e32

class Stock():

    def __init__(self, code, key, market=1, db_file_name='stocks.db', log_path_name='./logs/'):
        
        self.code = code
        self.key = key
        self.market = market
        self.db_file_name = db_file_name
        self.log_path_name = log_path_name
        self.columns = ['DateTime', 'OverSellQty', 'UnderBuyQty', 'TotalMarketValue', 'MarketOrderSellQty', 'MarketOrderBuyQty', 'BidTime', 'AskTime', 'Exchange', 'ExchangeName', 'TradingVolume', 'TradingVolumeTime', 'VWAP', 'TradingValue', 'BidQty', 'BidPrice', 'BidSign', 'AskQty', 'AskPrice', 'AskSign', 'Symbol', 'SymbolName', 'CurrentPrice', 'CurrentPriceTime', 'CurrentPriceChangeStatus', 'CurrentPriceStatus', 'CalcPrice', 'PreviousClose', 'PreviousCloseTime', 'ChangePreviousClose', 'ChangePreviousClosePer', 'OpeningPrice', 'OpeningPriceTime', 'HighPrice', 'HighPriceTime', 'LowPrice', 'LowPriceTime', 'SecurityType', 'Sell1_Price', 'Sell1_Qty', 'Sell1_Sign', 'Sell1_Time', 'Sell2_Price', 'Sell2_Qty', 'Sell3_Price', 'Sell3_Qty', 'Sell4_Price', 'Sell4_Qty', 'Sell5_Price', 'Sell5_Qty', 'Sell6_Price', 'Sell6_Qty', 'Sell7_Price', 'Sell7_Qty', 'Sell8_Price', 'Sell8_Qty', 'Sell9_Price', 'Sell9_Qty', 'Sell10_Price', 'Sell10_Qty', 'Buy1_Price', 'Buy1_Qty', 'Buy1_Sign', 'Buy1_Time', 'Buy2_Price', 'Buy2_Qty', 'Buy3_Price', 'Buy3_Qty', 'Buy4_Price', 'Buy4_Qty', 'Buy5_Price', 'Buy5_Qty', 'Buy6_Price', 'Buy6_Qty', 'Buy7_Price', 'Buy7_Qty', 'Buy8_Price', 'Buy8_Qty', 'Buy9_Price', 'Buy9_Qty', 'Buy10_Price', 'Buy10_Qty',]
        self.data = pd.DataFrame(index = [], columns = self.columns)
        
        
    def set_brand_data(self):
        
        content = self.key.fetch_brand_info(self.code, self.market)
        try:
            self.disp_name = content["DisplayName"]
            self.unit = int(content["TradingUnit"])
        except KeyError:
            exit('\033[31m銘柄略称・売買単位を取得できませんでした。\033[0m')
        except Exception:
            exit('\033[31m不明な例外により強制終了します。\033[0m')
        

    def fetch_data(self):
        
        # この銘柄の生データ（板情報）を新しく取得し、data の末尾に追加する
        board_info = self.key.fetch_board_info(self.code, self.market)
        df1 = pd.DataFrame([datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')], columns = ['DateTime'])
        df2 = pd.json_normalize(board_info, sep = '_')
        new_data = pd.concat([df1, df2], axis = 1)
        self.data = pd.concat([self.data, new_data])

        
    def register_to_stock_list(self):
        
        # この銘柄を登録銘柄リストに登録する
        content = self.key.push_register_request(self.code, self.market)
        
        
    def unregister_from_stock_list(self):
        
        # この銘柄を登録銘柄リストから削除する
        content = self.key.push_unregister_request(self.code, self.market)


    def store_daily_candle(self):

        # 日足のローソク足をデータベースに保存する

        # 各種の値段を計算する
        open_price = self.data['CurrentPrice'].head(1)[0]  # 始値
        highest_price = max(self.data['CurrentPrice']) # 最高値
        lowest_price = min(self.data['CurrentPrice'])  # 最安値
        close_price = self.data['CurrentPrice'].tail(1)[0]  # 終値

        # 取引量を計算する
        volume = self.data['TradingVolume'].tail(1)[0]
        
        # 現在日を取得する
        JST = timezone(timedelta(hours=+9), 'JST')
        index_date = datetime.now(JST).strftime('%Y-%m-%d')

        columns = ['code', 'uuid', 'date', 'open', 'high', 'low', 'close', 'volume']
        daily_candle_data = pd.DataFrame(index = [], columns = columns)
        daily_candle_data.loc[index_date] = {'code': self.code,
                                             'uuid': str(uuid.uuid4()),
                                             'date': index_date,
                                             'open': open_price,
                                             'high': highest_price,
                                             'low': lowest_price,
                                             'close': close_price,
                                             'volume': volume,}
        
        # 日足データをDBに格納する
        conn = sqlite3.connect(self.db_file_name)
        with conn:
            daily_candle_data.to_sql('DailyCandle', conn, if_exists='append', index=None)
