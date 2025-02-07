import os
import pickle
from datetime import datetime

import pandas as pd
pd.set_option('display.max_rows', None)

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class ModelLibrary:

    def __init__(self, n_symbols):
        
        self.n_symbols = n_symbols
        self.data = []
        for _ in range(n_symbols):
            self.data.append([])


    def append_data(self, new_data, index):

        self.data[index].append(new_data)


    def save_data(self):

        now = datetime.now()
        filename = now.strftime("data_%Y%m%d_%H%M%S.pkl")
        filepath = os.path.join("./", filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.data, f)


    def set_data(self, data):

        self.data = data
        
        
    def prepare_training_data(self):

        ## 入力データを準備する

        # 元のデータを用意する
        input_data = self.prepare_original_data()

        # 移動平均を計算する
        input_data = self.calc_moving_average(input_data)

        # MACDを計算する
        input_data = self.calc_macd(input_data)

        # ボリンジャーバンドを計算する
        input_data = self.calc_bollinger_band(input_data)

        # 一目均衡表を計算する
        input_data = self.calc_ichimoku(input_data)

        # RSIを計算する
        input_data = self.calc_rsi(input_data)

        breakpoint()


    # 元のデータを用意する
    def prepare_original_data(self):
        
        price_list = []
        volume_list = []
        for i in range(self.n_symbols):
            price_list.append([])
            volume_list.append([])
            for d in self.data[i]:
                if d['CurrentPriceTime'] is None:
                    continue
                try:
                    dt_object = datetime.fromisoformat(d['CurrentPriceTime'].replace('Z', '+00:00'))
                    formatted_datetime = dt_object.strftime("%Y-%m-%d %H:%M")
                except ValueError:
                    print("文字列のフォーマットが異なります。")
                    exit()
                price_list[i].append([formatted_datetime, d['CurrentPrice']])
                volume_list[i].append([formatted_datetime, d['TradingVolume']])

        price_data = []
        volume_data = []
        original_data = []
        for i in range(self.n_symbols):
            
            price_data.append(pd.DataFrame(price_list[i], columns = ['DateTime', 'Price']))
            price_data[i] = price_data[i].set_index('DateTime')
            price_data[i].index = pd.to_datetime(price_data[i].index)
            price_data[i] = price_data[i].resample('1Min').ohlc().dropna()
            price_data[i].columns = price_data[i].columns.get_level_values(1)
            
            volume_data.append(pd.DataFrame(volume_list[i], columns = ['DateTime', 'Volume']))
            volume_data[i].drop_duplicates(subset = 'DateTime', keep = 'first', inplace = True)
            volume_data[i] = volume_data[i].set_index('DateTime')
            volume_data[i].index = pd.to_datetime(volume_data[i].index)
            volume_data[i]['Volume'] = volume_data[i]['Volume'].diff(1).fillna(volume_data[i]['Volume'].iloc[0])

            original_data.append(pd.concat([price_data[i], volume_data[i]], axis = 1))
            
        return original_data


    def calc_moving_average(self, original_data):

        for i in range(self.n_symbols):
            original_data[i]['MA5'] = original_data[i]['close'].rolling(window=5).mean()
            original_data[i]['MA25'] = original_data[i]['close'].rolling(window=25).mean()

        return original_data


    def calc_macd(self, original_data):

        for i in range(self.n_symbols):
            original_data[i]['MACD'] = original_data[i]['close'].ewm(span=12).mean() - original_data[i]['close'].ewm(span=26).mean()
            original_data[i]['SIGNAL'] = original_data[i]['MACD'].ewm(span=9).mean()
            original_data[i]['HISTOGRAM'] = original_data[i]['MACD'] - original_data[i]['SIGNAL']

        return original_data


    def calc_bollinger_band(self, original_data):
        
        for i in range(self.n_symbols):
            sma20 = original_data[i]['close'].rolling(window=20).mean()
            std20 = original_data[i]['close'].rolling(window=20).std()
            original_data[i]['Upper'] = sma20 + (std20 * 2)
            original_data[i]['Lower'] = sma20 - (std20 * 2)
            
        return original_data


    def calc_ichimoku(self, original_data):

        for i in range(self.n_symbols):
            original_data[i]['ConversionLine'] = (original_data[i]['high'].rolling(window=9).max() + original_data[i]['low'].rolling(window=9).min()) / 2
            original_data[i]['BaseLine'] = (original_data[i]['high'].rolling(window=26).max() + original_data[i]['low'].rolling(window=26).min()) / 2
            original_data[i]['LeadingSpanA'] = ((original_data[i]['ConversionLine'] + original_data[i]['BaseLine']) / 2).shift(26)
            original_data[i]['LeadingSpanB'] = ((original_data[i]['high'].rolling(window=52).max() + original_data[i]['low'].rolling(window=52).min()) / 2).shift(26)
            original_data[i]['LaggingSpan'] = original_data[i]['close'].shift(-26)
            
        return original_data


    def calc_rsi(self, original_data):
        
        for i in range(self.n_symbols):
            delta = original_data[i]['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            original_data[i]['RSI'] = 100 - (100 / (1 + rs))
            
        return original_data
