import pandas as pd

class Stock:

    def __init__(self, symbol, lib, model, exchange=1):

        self.symbol = symbol
        self.lib= lib
        self.model = model
        self.exchange = exchange
        self.data = pd.DataFrame(index = [], columns = self.columns)

        
    def set_infomation(self):
        
        content = self.lib.fetch_information(self.symbol, self.exchange)
        try:
            self.disp_name = content["DisplayName"]
            self.unit = int(content["TradingUnit"])
        except KeyError:
            exit('\033[31m銘柄略称・売買単位を取得できませんでした。\033[0m')
        except Exception:
            exit('\033[31m不明な例外により強制終了します。\033[0m')
        
        
