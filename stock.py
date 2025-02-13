from datetime import datetime
import pandas as pd
import numpy as np


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
        
        if new_data['CurrentPriceTime'] is not None:
            dt_object = datetime.fromisoformat(new_data['CurrentPriceTime'].replace('Z', '+00:00'))
            self.time.append(dt_object.strftime("%Y-%m-%d %H:%M"))
            self.price.append(new_data['CurrentPrice'])
            self.volume.append(new_data['TradingVolume'])
        

    def prepare_data(self):

        price_df = pd.DataFrame({'DateTime': self.time, 'Price': self.price})
        price_df = price_df.set_index('DateTime')
        price_df.index = pd.to_datetime(price_df.index)
        price_df = price_df.resample('1Min').ohlc().dropna()  # 1分足に変換
        price_df.columns = price_df.columns.get_level_values(1)
        
        volume_df = pd.DataFrame({'DateTime': self.time, 'volume': self.volume})
        volume_df.drop_duplicates(subset = 'DateTime', keep = 'first', inplace = True)
        volume_df = volume_df.set_index('DateTime')
        volume_df.index = pd.to_datetime(volume_df.index)
        
        self.data = pd.concat([self.data, pd.concat([price_df, volume_df], axis = 1)])
        
        # 移動平均を計算する
        self.data = self.model.calc_moving_average(self.data)
        
        # MACDを計算する
        self.data = self.model.calc_macd(self.data)
        
        # ボリンジャーバンドを計算する
        self.data = self.model.calc_bollinger_band(self.data)
        
        # RSIを計算する
        self.data = self.model.calc_rsi(self.data)
        

    def predict(self):

        tmp = self.data.tail(self.window)
        if np.nan in tmp.values:
            return False     # データにNaNが含まれていたら何もしない
        elif len(tmp) < self.window:
            return False     # データが足らない場合も何もしない
        else:
            input_data = pd.DataFrame([tmp.values.reshape(-1)])
            print(f"\033[31m{input_data}\033[0m")
            predict_value = self.model.predict(input_data)
            print(predict_value)
            result = False if predict_value < 0.7 else True
            return result
            
    
    def polling(self):

        # １分間隔で呼ばれる関数
        if self.time is not None:

            # データを準備する
            self.prepare_data()
            
            print(f"\033[32m{self.symbol}：{self.disp_name}\033[0m")
            print(f"\033[32m{self.data}\033[0m")
            
            # 株価が上がるか否かを予測する
            result = self.predict()
            
            
        self.time = []
        self.price = []
        self.volume = []
                
