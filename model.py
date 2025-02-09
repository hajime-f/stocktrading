import os
import pickle
from datetime import datetime

import pandas as pd
pd.set_option('display.max_rows', None)

import numpy as np
from sklearn.utils import all_estimators
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import warnings

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class ModelLibrary:

    def __init__(self, n_symbols):
        
        self.n_symbols = n_symbols
        self.data = []
        for _ in range(n_symbols):
            self.data.append([])
        self.model = None


    def append_data(self, new_data, index):

        data ={'CurrentPriceTime': new_data['CurrentPriceTime'],
               'CurrentPrice': new_data['CurrentPrice'],
               'TradingVolume': new_data['TradingVolume']}
        self.data[index].append(data)


    def save_data(self):

        now = datetime.now()
        filename = now.strftime("data_%Y%m%d_%H%M%S.pkl")
        filename = os.path.join("./", filename)
        
        with open(filename, 'wb') as f:
            pickle.dump(self.data, f)

        return filename


    def set_data(self, p_data):

        concat_data = []

        for d in p_data:
            concat_data += d

        self.data = concat_data
        
        
    def prepare_raw_data(self):

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
        # input_data = self.calc_ichimoku(input_data)

        # RSIを計算する
        input_data = self.calc_rsi(input_data)

        # ストキャスティクスを計算する
        # input_data = self.calc_stochastic(input_data)
        
        ## 出力データを準備する        
        output_data = self.calc_output_data(input_data)

        ## 入力データと出力データを結合して学習用の生データを作成する
        raw_data = []
        for i in range(self.n_symbols):
            raw_data.append(self.concat_dataframes(input_data[i], output_data[i]).dropna())

        return raw_data
        

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
            price_data[i] = price_data[i].resample('1Min').ohlc().dropna()  # 1分足に変換
            price_data[i].columns = price_data[i].columns.get_level_values(1)

            volume_data.append(pd.DataFrame(volume_list[i], columns = ['DateTime', 'volume']))
            volume_data[i].drop_duplicates(subset = 'DateTime', keep = 'first', inplace = True)
            volume_data[i] = volume_data[i].set_index('DateTime')
            volume_data[i].index = pd.to_datetime(volume_data[i].index)
            volume_data[i]['volume'] = volume_data[i]['volume'].diff(1).fillna(volume_data[i]['volume'].iloc[0])

            original_data.append(pd.concat([price_data[i], volume_data[i]], axis = 1))
            
        return original_data


    def calc_moving_average(self, data):

        for i in range(self.n_symbols):
            data[i]['MA5'] = data[i]['close'].rolling(window=5).mean()
            data[i]['MA25'] = data[i]['close'].rolling(window=25).mean()

        return data


    def calc_macd(self, data):

        for i in range(self.n_symbols):
            data[i]['MACD'] = data[i]['close'].ewm(span=12).mean() - data[i]['close'].ewm(span=26).mean()
            data[i]['SIGNAL'] = data[i]['MACD'].ewm(span=9).mean()
            data[i]['HISTOGRAM'] = data[i]['MACD'] - data[i]['SIGNAL']

        return data


    def calc_bollinger_band(self, data):
        
        for i in range(self.n_symbols):
            sma20 = data[i]['close'].rolling(window=20).mean()
            std20 = data[i]['close'].rolling(window=20).std()
            data[i]['Upper'] = sma20 + (std20 * 2)
            data[i]['Lower'] = sma20 - (std20 * 2)
            
        return data


    def calc_ichimoku(self, data):

        for i in range(self.n_symbols):
            data[i]['ConversionLine'] = (data[i]['high'].rolling(window=9).max() + data[i]['low'].rolling(window=9).min()) / 2
            data[i]['BaseLine'] = (data[i]['high'].rolling(window=26).max() + data[i]['low'].rolling(window=26).min()) / 2
            data[i]['LeadingSpanA'] = ((data[i]['ConversionLine'] + data[i]['BaseLine']) / 2).shift(26)
            data[i]['LeadingSpanB'] = ((data[i]['high'].rolling(window=52).max() + data[i]['low'].rolling(window=52).min()) / 2).shift(26)
            data[i]['LaggingSpan'] = data[i]['close'].shift(-26)
            
        return data


    def calc_rsi(self, data):
        
        for i in range(self.n_symbols):
            delta = data[i]['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data[i]['RSI'] = 100 - (100 / (1 + rs))
            
        return data


    def calc_stochastic(self, data):
        
        for i in range(self.n_symbols):
            data[i]['%K'] = 100 * (data[i]['close'] - data[i]['low'].rolling(window=14).min()) / (data[i]['high'].rolling(window=14).max() - data[i]['low'].rolling(window=14).min())
            data[i]['%D'] = data[i]['%K'].rolling(window=3).mean()
            
        return data


    def calc_output_data(self, input_data):

        output_data = []
        
        for i in range(self.n_symbols):
            close_price = input_data[i]['close']
            tmp1 = self.check_price_change(close_price, 0.5)
            # tmp2 = self.check_price_change(close_price, -0.5)
            # output_data.append(tmp1 + tmp2)
            output_data.append(tmp1)

        return output_data
    

    def check_price_change(self, stock_price, percentage, time_window = 20):

        # ある時刻における株価を基準にして、そこからtime_window分以内にpercentage％変化するか否かを判定する。
        
        result = []
        
        for i in range(len(stock_price) - time_window):
            base_price = stock_price.iloc[i]  # 基準時刻の株価
            target_price = base_price * (1 + percentage / 100)  # 目標株価
            
            # 基準時刻からtime_window分後の株価を取得
            future_prices = stock_price[i + 1:i + time_window + 1]
            
            # time_window分以内に目標株価に達しているか確認
            if percentage > 0:                
                if (future_prices >= target_price).any():
                    result.append(1)
                else:
                    result.append(0)
            elif percentage < 0:
                if (future_prices <= target_price).any():
                    result.append(-1)
                else:
                    result.append(0)
            else:
                pass

        return pd.DataFrame(result, columns = ['Result'])
        
    
    def concat_dataframes(self, data1, data2):
        
        # 行数の多い方を取得
        max_rows = max(len(data1), len(data2))
        
        # 行数が少ない方にNaNを追加
        if len(data1) < max_rows:
            data1 = pd.concat([data1, pd.DataFrame([[pd.NA] * len(data1.columns)] * (max_rows - len(data1)), columns=data1.columns)], ignore_index=True)
            data1.index = data2.index
        elif len(data2) < max_rows:
            data2 = pd.concat([data2, pd.DataFrame([[pd.NA] * len(data2.columns)] * (max_rows - len(data2)), columns=data2.columns)], ignore_index=True)
            data2.index = data1.index
        else:
            pass
        
        # 2つのDataFrameを結合
        data_concat = pd.concat([data1, data2], axis = 1)
        return data_concat


    def prepare_training_data(self, raw_data, window = 10):

        X = pd.DataFrame()
        Y = pd.DataFrame()
        
        for r in raw_data:
            for i in range(len(r) - window):
                
                # tmp1 = r.drop(['open', 'high', 'low', 'Result'], axis = 1).iloc[i:i + window]
                tmp1 = r.drop(['Result'], axis = 1).iloc[i:i + window]
                tmp2 = r.Result.iloc[i + window - 1]

                X = pd.concat([X, pd.DataFrame([tmp1.values.reshape(-1)])])
                Y = pd.concat([Y, pd.DataFrame([tmp2])])

        X = X.reset_index(drop = True)
        Y = Y.reset_index(drop = True)

        return X, Y
    

    def evaluate_model(self, X, Y):

        # クロスバリデーション用のオブジェクトをインスタンス化する
        kfold_cv = KFold(n_splits=6, shuffle=False)
        warnings.filterwarnings('ignore')
        
        # classifier のアルゴリズムをすべて取得する
        all_Algorithms = all_estimators(type_filter="classifier")
        warnings.filterwarnings('ignore')
        
        max_clf = None
        max_score = -1

        # 面倒なモデルは全部スキップする
        negative_list = ["RadiusNeighborsClassifier", "CategoricalNB", "ClassifierChain", "ComplementNB", "FixedThresholdClassifier", "GaussianProcessClassifier", "GradientBoostingClassifier", "MLPClassifier", "MultiOutputClassifier", "MultinomialNB", "NuSVC", "OneVsOneClassifier", "OneVsRestClassifier", "OutputCodeClassifier", "StackingClassifier", "VotingClassifier", "TunedThresholdClassifierCV", "RadiusNeighborsClassifier", "LinearSVC"]
        
        # 各分類アルゴリズムをクロスバリデーションで評価する
        for (name, algorithm) in all_Algorithms:
            
            try:
                if name in negative_list:
                    continue
                
                clf = algorithm()
                    
                if hasattr(clf, "score"):
                    scores = cross_val_score(clf, X, Y, cv=kfold_cv)
                    m = round(np.mean(scores) * 100, 2)
                    print(name, "の正解率：", m, "％")
                    if max_score < m:
                        max_clf = clf
                        max_score = m
                        
            except Exception as e:
                print(e)

        return max_clf


    def validate_model(self, clf, X, Y):

        # データを学習用データとテスト用データに分割する
        n_train = int(len(X) * 0.8)
        X_train, Y_train = X[:n_train], Y[:n_train]
        X_test, Y_test = X[n_train:], Y[n_train:]

        # モデルを学習する
        clf = clf.fit(X_train, Y_train)

        # モデルを評価する
        score = clf.score(X_test, Y_test)
        print(f'正解率：{score}')

        return clf


    def save_model(self, model):

        now = datetime.now()
        filename = now.strftime("model_%Y%m%d_%H%M%S.pkl")
        filename = os.path.join("./", filename)
        
        with open(filename, 'wb') as f:
            pickle.dump(model, f)

        return filename


    def load_model(self, filename):

        with open(filename, 'rb') as f:
            self.model = pickle.load(f)

        return self.model
            
    
    def _debug_plot_graph(self, data):

        import mplfinance as mpf

        mpf.plot(data[['open', 'high', 'low', 'close', 'volume']], type='candle', volume=True, figratio=(12, 4), style='charles', savefig='debug.png')
    
