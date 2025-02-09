from datetime import datetime


class Stock:

    def __init__(self, symbol, lib, model, exchange = 1, window = 10):

        self.symbol = symbol
        self.lib= lib
        self.model = model
        self.exchange = exchange
        
        self.time = []
        self.price = []
        self.volume = []
        
        self.data = pd.DataFrame()
        self.window = window

        
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
        
        dt_object = datetime.fromisoformat(new_data['CurrentPriceTime'].replace('Z', '+00:00'))
        self.time.append(dt_object.strftime("%Y-%m-%d %H:%M"))
        self.price.append(new_data['CurrentPrice'])
        self.volume.append(new_data['TradingVolume'])
        

    def polling(self):

        # １分間隔で呼ばれる関数
        
        price_df = pd.DataFrame([self.time, self.price], columns = ['DateTime', 'Price'])
        price_df = price_df.set_index('DateTime')
        price_df.index = pd.to_datetime(price_df.index)
        price_df = price_df.resample('1Min').ohlc().dropna()  # 1分足に変換
        price_df.columns = price_df.columns.get_level_values(1)

        volume_df = pd.DataFrame([self.time, self.volume], columns = ['DateTime', 'volume'])
        volume_df.drop_duplicates(subset = 'DateTime', keep = 'first', inplace = True)
        volume_df = volume_df.set_index('DateTime')
        volume_df.index = pd.to_datetime(volume_df.index)

        self.data.concat(pd.concat([price_df, volume_df], axis = 1))

        self.time = []
        self.price = []
        self.volume = []
        
        
        
        
        
