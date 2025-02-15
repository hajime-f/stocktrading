import os
import pickle
from datetime import datetime

import pandas as pd
pd.set_option('display.max_rows', None)

import numpy as np
from sklearn.utils import all_estimators
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import warnings

import xgboost as xgb
from sklearn.model_selection import train_test_split


class ModelLibrary:

    def __init__(self, n_symbols):
        
        self.n_symbols = n_symbols
        self.data = []
        for _ in range(n_symbols):
            self.data.append([])
        self.clf = None


    def append_data(self, new_data, index):

        if new_data['CurrentPriceTime'] is not None:        
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


    def prepare_dataframe_list(self):

        # 生データを分足のDataFrameに変換する
        df_list = [self.convert_to_dataframe(d) for d in self.data]

        # データを正規化する
        df_list = [self.normalize_data(df) for df in df_list]

        # 移動平均を計算する
        df_list = [self.calc_moving_average(df) for df in df_list]

        # MACDを計算する
        df_list = [self.calc_macd(df) for df in df_list]

        # ボリンジャーバンドを計算する
        df_list = [self.calc_bollinger_band(df) for df in df_list]

        # RSIを計算する
        df_list = [self.calc_rsi(df) for df in df_list]

        # 正解ラベルを作成する
        label_list = [self.check_price_change(df['close'], 100) for df in df_list]

        # データを結合する
        XY = [self.concat_dataframes(input_df, label_df).dropna() for input_df, label_df in zip(df_list, label_list)]
        
        return XY

        
    def convert_to_dataframe(self, original_data):

        price_list = []
        
        for d in original_data:

            if d['CurrentPriceTime'] is None:
                continue
            
            dt_object = datetime.fromisoformat(d['CurrentPriceTime'].replace('Z', '+00:00'))
            formatted_datetime = dt_object.strftime("%Y-%m-%d %H:%M")            
            price_list.append([formatted_datetime, d['CurrentPrice']])

        price_df = pd.DataFrame(price_list, columns = ['DateTime', 'Price']).set_index('DateTime')
        price_df.index = pd.to_datetime(price_df.index)
        price_df = price_df.resample('1Min').ohlc().dropna()
        price_df.columns = price_df.columns.get_level_values(1)

        return price_df


    def normalize_data(self, data):

        max_value = data.iloc[0]['high']
        min_value = data.iloc[0]['low']

        return (data - min_value) / (max_value - min_value)        


    def calc_moving_average(self, data):

        data['MA5'] = data['close'].rolling(window=5).mean()
        data['MA25'] = data['close'].rolling(window=25).mean()

        return data


    def calc_macd(self, data):

        data['MACD'] = data['close'].ewm(span=12).mean() - data['close'].ewm(span=26).mean()
        data['SIGNAL'] = data['MACD'].ewm(span=9).mean()
        data['HISTOGRAM'] = data['MACD'] - data['SIGNAL']

        return data


    def calc_bollinger_band(self, data):
        
        sma20 = data['close'].rolling(window=20).mean()
        std20 = data['close'].rolling(window=20).std()
        data['Upper'] = sma20 + (std20 * 2)
        data['Lower'] = sma20 - (std20 * 2)
            
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
        
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
            
        return data


    def calc_stochastic(self, data):
        
        for i in range(self.n_symbols):
            data[i]['%K'] = 100 * (data[i]['close'] - data[i]['low'].rolling(window=14).min()) / (data[i]['high'].rolling(window=14).max() - data[i]['low'].rolling(window=14).min())
            data[i]['%D'] = data[i]['%K'].rolling(window=3).mean()
            
        return data
    

    def check_price_change(self, stock_price, percentage, time_window = 20):

        # ある時刻における株価を基準にして、そこからtime_window分以内にpercentage％変化するか否かを判定する。
        
        result = []
        
        if percentage == 0:
            raise ValueError("percentageは0以外の値を指定してください。")

        for i in range(len(stock_price) - time_window):

            base_price = stock_price.iloc[i]  # 基準時刻の株価

            # 基準株価が0の場合は何もしない
            if base_price == 0:
                result.append(0)
                continue
            
            target_price = base_price * (1 + percentage / 100)  # 目標株価

            # 基準時刻からtime_window分後の株価を取得
            end_index = min(i + time_window + 1, len(stock_price))
            future_prices = stock_price.iloc[i + 1:end_index]

            if base_price > 0:  # 株価が正の場合
                if (future_prices >= target_price).any():
                    result.append(1)  # 上昇
                else:
                    result.append(0)  # 上昇せず
            elif base_price < 0:  # 株価が負の場合
                if (future_prices <= target_price).any():
                    result.append(1)  # 上昇（絶対値は減少）
                else:
                    result.append(0)  # 上昇せず
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
        
        best_clf = None
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
                        best_clf = clf
                        max_score = m
                        
            except Exception as e:
                print(e)

        return best_clf


    def validate_model(self, clf, X, Y):

        # データをシャッフルする
        X_shuffled, Y_shuffled = shuffle(X, Y, random_state = 42)
        
        # データを学習用データとテスト用データに分割する
        n_train = int(len(X_shuffled) * 0.8)
        X_train, Y_train = X_shuffled[:n_train], Y_shuffled[:n_train]
        X_test, Y_test = X_shuffled[n_train:], Y_shuffled[n_train:]

        # モデルを学習する
        clf = clf.fit(X_train, Y_train)

        # モデルを評価する
        score = clf.score(X_test, Y_test)
        print(f'正解率：{score}')

        Y_pred = clf.predict(X_test)
        print(classification_report(Y_test, Y_pred))

        return clf


    def evaluate_xgboost(self, X, Y):

        # データを学習用データとテスト用データに分割する
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # データをDMatrix形式に変換する
        dtrain = xgb.DMatrix(X_train, label=Y_train)
        dtest = xgb.DMatrix(X_test, label=Y_test)

        # パラメータを設定する
        param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
        param['nthread'] = 4
        param['eval_metric'] = 'logloss'

        # モデルを学習する
        evallist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = 10
        bst = xgb.train(param, dtrain, num_round, evallist)

        # モデルを評価する
        preds = bst.predict(dtest)
        labels = dtest.get_label()
        print(classification_report(labels, preds > 0.5))        

        return bst
    
    
    def save_model(self, model):

        now = datetime.now()
        filename = now.strftime("model_%Y%m%d_%H%M%S.pkl")
        filename = os.path.join("./", filename)
        
        with open(filename, 'wb') as f:
            pickle.dump(model, f)

        return filename


    def load_model(self, filename):

        with open(filename, 'rb') as f:
            self.clf = pickle.load(f)

        return self.clf
            
    
    def predict(self, data):

        return self.clf.predict(data)
                    
