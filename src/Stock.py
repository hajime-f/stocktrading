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
        
        
    def register_to_stock_list(self):
        
        # この銘柄を登録銘柄リストに登録する
        content = self.key.push_register_request(self.code, self.market)
        
        
    def unregister_from_stock_list(self):
        
        # この銘柄を登録銘柄リストから削除する
        content = self.key.push_unregister_request(self.code, self.market)
        




