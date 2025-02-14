from datetime import datetime, timedelta, time
import pandas as pd
import numpy as np


class Stock:

    def __init__(self, symbol, lib, model, base_transaction, exchange = 1, window = 10):

        self.symbol = symbol
        self.lib= lib
        self.model = model
        self.base_transaction = base_transaction
        self.exchange = exchange
        
        self.time = []
        self.price = []
        self.volume = []
        
        self.data = pd.DataFrame()
        self.window = window

        self.buy_order_flag = False
        self.buy_order_id = None
        self.sell_order_flag = False
        self.sell_order_id = None
        self.purchase_price = 0

        
    def set_infomation(self):
        
        content = self.lib.fetch_information(self.symbol, self.exchange)
        try:
            self.disp_name = content["DisplayName"]
            self.unit = int(content["TradingUnit"])
            self.transaction_unit = self.unit * self.base_transaction
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
        raw_data = self.data.copy()
        
        # 移動平均を計算する
        raw_data = self.model.calc_moving_average(raw_data)
        
        # MACDを計算する
        raw_data = self.model.calc_macd(raw_data)
        
        # ボリンジャーバンドを計算する
        raw_data = self.model.calc_bollinger_band(raw_data)
        
        # RSIを計算する
        raw_data = self.model.calc_rsi(raw_data)

        # # 重複行を削除する
        # if raw_data.index is not None:
        #     raw_data.drop_duplicates(subset=raw_data.index.name, keep = 'last')

        print(f"\033[32m{self.symbol}：{self.disp_name}\033[0m")
        print(f"\033[32m{raw_data}\033[0m")
            
        return raw_data
        

    def predict(self, raw_data):

        tmp = raw_data.tail(self.window)
        if len(tmp) < self.window:
            return False     # データが足らない場合は何もしない
        elif tmp.isnull().values.any():
            return False     # データにNaNが含まれている場合も何もしない
        else:
            input_data = tmp.values.reshape(-1)
            print(f"\033[31m{input_data}\033[0m")
            predict_value = self.model.predict([input_data])
            print(predict_value)
            result = False if predict_value < 0.7 else True
            return result
            
    
    def polling(self):

        ## １分間隔で呼ばれる関数
        
        # 買い注文が約定したか否かをチェックし、約定している場合はフラグを更新する
        if self.buy_order_flag:
            self.check_and_update_buy_order_status()
                
        # 売り注文が約定したか否かをチェックし、約定している場合はフラグを更新する
        if self.sell_order_flag:
            self.check_and_update_sell_order_status()

            # 売り注文が残っている場合は時価が買った時の値段を下回っていないか否かをチェックする
            
                
        if self.time is not None:

            # データを準備する
            raw_data = self.prepare_data()
            
            # 株価が上がるか否かを予測する
            result = self.predict(raw_data)

            # 上がると予測された場合
            if result:

                # 取引を実行する
                result = self.execute_transaction()
                
                
            
        self.time = []
        self.price = []
        self.volume = []


    def check_and_update_buy_order_status(self):
        pass


    def check_and_update_sell_order_status(self):
        pass
        

    def execute_transaction(self):
        
        # 取引価格を計算する
        price = self.lib.fetch_price(self.symbol, self.exchange)
        transaction_price = price * self.transaction_unit
        print(f"\033[34m取引価格：{int(transaction_price):,} 円\033[0m")
        
        # 買付余力が取引価格を上回っている（買える）場合
        if self.lib.deposit() > transaction_price:

            # まだ売り注文が残っている場合は買わない
            if self.sell_order_flag:
                print(f"\033[32m値上がりが予測され、買付余力もありましたが、すでにこの銘柄を買っているので発注しませんでした。\033[0m")
                return False

            # 15:30まで20分を切っている場合は買わない
            now = datetime.now()
            target_time = datetime.combine(now.date(), time(15, 30))
            time_limit = target_time - timedelta(minutes=20)
            if now < time_limit:
                print(f"\033[32m値上がりが予測され、買付余力もありましたが、15:30まで20分を切っているので発注しませんでした。\033[0m")
                return False
            
            # 成行で買い注文を入れる
            content = buy_at_market_price_with_cash(self.symbol, self.transaction_unit, self.exchange)
            order_result1 = content['Result']
            order_id1 = content['OrderId']
            if order_result1 == 0:
                self.buy_order_flag = True
                self.buy_order_id = order_id1
                self.purchase_price = price
                
                # 指値で売り注文を入れる
                content = sell_at_limit_price(self.symbol, self.transaction_unit, self.purchase_price * 1.05, self.exchange)
                order_result2 = content['Result']
                order_id2 = content['OrderId']
                if order_result2 == 0:
                    self.sell_order_flag = True
                    self.sell_order_id = order_id2

            if order_result1 == 0 and order_result2 == 0:
                print(f"\033[32m{self.symbol}：{self.disp_name} の買い注文・売り注文が正常に発注されました。\033[0m")
                return True
            else:
                print(f"\033[32m値上がりが予測され、買付余力もありましたが、なんらかの原因により発注できませんでした。\033[0m")
                return False

        else:
            print(f"\033[32m値上がりが予測されましたが、買付余力がありませんでした。\033[0m")
            return False
            
