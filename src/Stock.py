import pandas as pd
import sqlite3, uuid
from datetime import date
from datetime import datetime

class Stock():

    def __init__(self, code, key, market=1, db_file_name='stocks.db', log_path_name='./logs/'):

        self.code = code
        self.key = key
        self.market = market
        self.db_file_name = db_file_name
        self.log_path_name = log_path_name

        content = key.fetch_brand_info(code, 1)
        try:
            self.disp_name = content["DisplayName"]
            self.unit = int(content["TradingUnit"])
        except KeyError:
            exit('\033[31m銘柄略称・売買単位を取得できませんでした。\033[0m')
        except Exception:
            exit('\033[31m不明な例外により強制終了します。\033[0m')

        self.column = ['DateTime', 'OverSellQty', 'UnderBuyQty', 'TotalMarketValue', 'MarketOrderSellQty', 'MarketOrderBuyQty', 'BidTime', 'AskTime', 'Exchange', 'ExchangeName', 'TradingVolume', 'TradingVolumeTime', 'VWAP', 'TradingValue', 'BidQty', 'BidPrice', 'BidSign', 'AskQty', 'AskPrice', 'AskSign', 'Symbol', 'SymbolName', 'CurrentPrice', 'CurrentPriceTime', 'CurrentPriceChangeStatus', 'CurrentPriceStatus', 'CalcPrice', 'PreviousClose', 'PreviousCloseTime', 'ChangePreviousClose', 'ChangePreviousClosePer', 'OpeningPrice', 'OpeningPriceTime', 'HighPrice', 'HighPriceTime', 'LowPrice', 'LowPriceTime', 'SecurityType', 'Sell1_Price', 'Sell1_Qty', 'Sell1_Sign', 'Sell1_Time', 'Sell2_Price', 'Sell2_Qty', 'Sell3_Price', 'Sell3_Qty', 'Sell4_Price', 'Sell4_Qty', 'Sell5_Price', 'Sell5_Qty', 'Sell6_Price', 'Sell6_Qty', 'Sell7_Price', 'Sell7_Qty', 'Sell8_Price', 'Sell8_Qty', 'Sell9_Price', 'Sell9_Qty', 'Sell10_Price', 'Sell10_Qty', 'Buy1_Price', 'Buy1_Qty', 'Buy1_Sign', 'Buy1_Time', 'Buy2_Price', 'Buy2_Qty', 'Buy3_Price', 'Buy3_Qty', 'Buy4_Price', 'Buy4_Qty', 'Buy5_Price', 'Buy5_Qty', 'Buy6_Price', 'Buy6_Qty', 'Buy7_Price', 'Buy7_Qty', 'Buy8_Price', 'Buy8_Qty', 'Buy9_Price', 'Buy9_Qty', 'Buy10_Price', 'Buy10_Qty',]
        self.data = pd.DataFrame(index=[], columns=self.column)
        

    def fetch_data(self):
        
        # この銘柄の情報を新しく取得し、data の末尾に追加する
        board_info = self.key.fetch_board_info(self.code, self.market)
        df1 = pd.DataFrame([datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')], columns=['DateTime'])
        df2 = pd.json_normalize(board_info, sep='_')
        new_data = pd.concat([df1, df2], axis=1)
        self.data = pd.concat([self.data, new_data])

        
    def register_to_stock_list(self):
        
        # この銘柄を登録銘柄リストに登録する
        content = self.key.push_register_request(self.code, self.market)
        
        
    def unregister_from_stock_list(self):
        
        # この銘柄を登録銘柄リストから削除する
        content = self.key.push_unregister_request(self.code, self.market)
        




