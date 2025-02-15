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
        self.loss_cut = False

        
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

        price_df = pd.DataFrame({'DateTime': self.time, 'Price': self.price}).set_index('DateTime')
        price_df.index = pd.to_datetime(price_df.index)
        price_df = price_df.resample('1Min').ohlc().dropna()
        price_df.columns = price_df.columns.get_level_values(1)

        self.data = pd.concat([self.data, price_df])
        raw_data = self.data.copy()

        # データを正規化する
        raw_data = self.model.normalize_data(raw_data)
        
        # 移動平均を計算する
        raw_data = self.model.calc_moving_average(raw_data)
        
        # MACDを計算する
        raw_data = self.model.calc_macd(raw_data)
        
        # ボリンジャーバンドを計算する
        raw_data = self.model.calc_bollinger_band(raw_data)
        
        # RSIを計算する
        raw_data = self.model.calc_rsi(raw_data)
            
        return raw_data
        

    def predict(self, raw_data):

        tmp = raw_data.tail(self.window)
        if len(tmp) < self.window:
            return False     # データが足らない場合は何もしない
        elif tmp.isnull().values.any():
            return False     # データにNaNが含まれている場合も何もしない
        else:
            input_data = tmp.values.reshape(-1)
            predict_value = self.model.predict([input_data])
            result = False if predict_value < 0.7 else True
            return result
            
    
    def polling(self):

        ## １分間隔で呼ばれる関数
        
        if self.buy_order_flag:

            # 買い注文が約定したか否かをチェックし、約定している場合はフラグを更新する
            if self.check_and_update_buy_order_status():
                
                # 指値で売り注文を出す
                sell_result = self.sell_at_limit_price()
                        
        if self.sell_order_flag:

            # 売り注文が約定したか否かをチェックし、約定している場合はフラグを更新する
            if not self.check_and_update_sell_order_status():

                # 売り注文が残っているとき、以下のいずれかの条件を満たす場合は成行で売り注文を出す（ロスカット）
                # (1) 時価が買った時の値段の95％を下回っている
                # (2) 15:30まで2分を切っている
                cond_result = self.conditional_market_sell()
            
        if self.time is not None:
            
            # データを準備する
            raw_data = self.prepare_data()
            print(f"\033[32mデータを更新しました：\033[0m{self.disp_name}（{self.symbol}）")
            
            # 株価が上がるか否かを予測する
            predict_result = self.predict(raw_data)
            
            # 上がると予測された場合
            if predict_result:
                
                # 成行で買い注文を出す
                buy_result = self.buy_at_market_price_with_cash()
            
                
            
        self.time = []
        self.price = []
        self.volume = []


    def conditional_market_sell(self):

        # すでにロスカット中の場合は何もしない
        if self.loss_cut:
            return False

        # 時価が買った時の値段の95％を下回っているか否かをチェックする
        price = self.lib.fetch_price(self.symbol, self.exchange)
        condition1 = price < self.purchase_price * 0.95

        # 15:30まで10分を切っているか否かをチェックする
        now = datetime.now()
        target_time = datetime.combine(now.date(), time(15, 30))
        time_limit = target_time - timedelta(minutes = 2)
        condition2 = now < time_limit
        
        if condition1 or condition2:
            
            # 先の売り注文をキャンセルする
            self.lib.cancel_order(self.sell_order_id)
            
            # 成行で売り注文を出す
            content = self.lib.sell_at_market_price(self.symbol, self.transaction_unit, self.exchange)
            order_result = content['Result']
            if order_result == 0:
                self.sell_order_flag = True
                self.loss_cut = True
                self.sell_order_id = content['OrderId']
                print(f"\033[34m成行で売り注文を出しました（ロスカット）。\033[0m")
                return True
            else:
                print(f"\033[34m条件により売り注文を出せませんでした。\033[0m")
                return False

        return False
            
        
    def check_and_update_buy_order_status(self):

        # 買い注文の約定状況を確認する
        result = self.lib.check_execution(self.buy_order_id)

        # 約定している場合
        if result['OrderState'] == 5:
            
            self.buy_order_flag = False
            self.purchase_price = result['Price']
            print(f"\033[33m{self.disp_name}（{self.symbol}）を {self.purchase_price:,} 円で購入しました。\033[0m")

            return True

        return False


    def check_and_update_sell_order_status(self):

        # 売り注文の約定状況を確認する
        result = self.lib.check_execution(self.sell_order_id)

        # 約定している場合
        if result['OrderState'] == 5:

            self.sell_order_flag = False
            self.loss_cut = False
            print(f"\033[33m{self.disp_name}（{self.symbol}）を {result['Price']:,} 円で売却しました（損益：{(result['Price'] - self.purchase_price) * self.transaction_unit:,}）。\033[0m")
            self.purchase_price = 0

            return True

        return False
        

    def sell_at_limit_price(self):

        # 指値で売り注文を出す
        content = self.lib.sell_at_limit_price(self.symbol, self.transaction_unit, self.purchase_price * 1.005, self.exchange)
        order_result = content['Result']
        if order_result == 0:
            self.sell_order_flag = True
            self.sell_order_id = content['OrderId']
            print(f"\033[34m売り注文を出しました。\033[0m")
            return True
        else:
            print(f"\033[34m購入した株を売る注文を出せませんでした。\033[0m")
            return False
    
    
    def buy_at_market_price_with_cash(self):

        # まだ売り注文が残っている場合は買わない
        if self.sell_order_flag:
            print(f"\033[34m値上がりが予測されましたが、すでにこの銘柄を買っているので発注しませんでした。\033[0m")
            return False
        
        # 15:30まで20分を切っている場合は買わない
        now = datetime.now()
        target_time = datetime.combine(now.date(), time(15, 30))
        time_limit = target_time - timedelta(minutes = 20)
        if now < time_limit:
            print(f"\033[34m値上がりが予測されましたが、15:30まで20分を切っているので発注しませんでした。\033[0m")
            return False
        
        # 取引価格を計算する
        price = self.lib.fetch_price(self.symbol, self.exchange)
        transaction_price = price * self.transaction_unit
        
        # 買付余力が取引価格を上回っている（買える）場合
        if self.lib.deposit() > transaction_price:

            # 成行で買い注文を入れる
            content = self.lib.buy_at_market_price_with_cash(self.symbol, self.transaction_unit, self.exchange)
            order_result = content['Result']
            if order_result == 0:
                self.buy_order_flag = True
                self.buy_order_id = content['OrderId']
                print(f"\033[34m買い注文を出しました。\033[0m")
                return True
            else:
                print(f"\033[34m値上がりが予測され、買付余力もありましたが、なんらかの原因により発注できませんでした。\033[0m")
                return False                            
            
        else:
            print(f"\033[34m値上がりが予測されましたが、買付余力がありませんでした。\033[0m")
            return False

    
