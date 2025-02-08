class Stock:

    def __init__(self, symbol, lib, model, exchange=1):

        self.symbol = symbol
        self.lib= lib
        self.model = model
        self.exchange = exchange
        self.data = []

        
    def set_infomation(self):
        
        content = self.lib.fetch_information(self.symbol, self.exchange)
        try:
            self.disp_name = content["DisplayName"]
            self.unit = int(content["TradingUnit"])
        except KeyError:
            exit('\033[31m銘柄略称・売買単位を取得できませんでした。\033[0m')
        except Exception:
            exit('\033[31m不明な例外により強制終了します。\033[0m')
        
        
    def append_data(self, new_data):

        data ={'CurrentPriceTime': new_data['CurrentPriceTime'],
               'CurrentPrice': new_data['CurrentPrice'],
               'TradingVolume': new_data['TradingVolume']}
        self.data.append(data)


    def polling(self):
        pass
