import os
import pickle
from datetime import datetime
import pandas as pd

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

        columns = ['DateTime', 'Price']
        self.training_data = []

        df_data = []
        for i in range(self.n_symbols):
            df_data.append([])
            for d in self.data[i]:
                if d['CurrentPriceTime'] is None:
                    continue
                try:
                    dt_object = datetime.fromisoformat(d['CurrentPriceTime'].replace('Z', '+00:00'))
                    formatted_datetime = dt_object.strftime("%Y-%m-%d %H:%M")
                except ValueError:
                    print("文字列のフォーマットが異なります。")
                    exit()
                df_data[i].append([formatted_datetime, d['CurrentPrice']])

        for i in range(self.n_symbols):
            self.training_data.append(pd.DataFrame(df_data[i], columns = columns))

        # 重複を落とす
        for t in self.training_data:
            t.drop_duplicates(subset = 'DateTime', keep = 'first', inplace = True)

        # -1〜1の間で正規化する
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = []
        for i in range(self.n_symbols):
            scaled_data.append(scaler.fit_transform(self.training_data[i][['Price']]))
            
        breakpoint()


