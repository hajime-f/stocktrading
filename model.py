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

        original_data = self.prepare_original_data()
        original_data = self.calc_moving_average(original_data)

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



    
