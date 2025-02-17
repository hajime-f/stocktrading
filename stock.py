from datetime import datetime, timedelta, time
import pandas as pd
import numpy as np
from rich.console import Console


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

        self.max_value = 0
        self.min_value = 0
        
        self.data = pd.DataFrame()
        self.window = window

        self.buy_order_flag = False
        self.buy_order_id = None
        self.sell_order_flag = False
        self.sell_order_id = None
        self.purchase_price = 0
        self.loss_cut = False

        self.console = Console()

        
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
        raw_data = self.normalize_data(raw_data)
        
        # 移動平均を計算する
        raw_data = self.model.calc_moving_average(raw_data)
        
        # MACDを計算する
        raw_data = self.model.calc_macd(raw_data)
        
        # ボリンジャーバンドを計算する
        raw_data = self.model.calc_bollinger_band(raw_data)
        
        # RSIを計算する
        raw_data = self.model.calc_rsi(raw_data)

        return raw_data


    def normalize_data(self, data):
        
        if self.max_value == 0 or self.min_value == 0:

            if len(data) >= 1:
                
                self.max_value = data.iloc[0]['high']
                self.min_value = data.iloc[0]['low']
                if self.max_value - self.min_value == 0:
                    self.max_value += 1
                return (data - self.min_value) / (self.max_value - self.min_value)

            return data

        else:

            return (data - self.min_value) / (self.max_value - self.min_value)
                

    def predict(self, raw_data):

        tmp = raw_data.tail(self.window)
        if len(tmp) < self.window:
            return False     # データが足らない場合は何もしない
        elif tmp.isnull().values.any():
            return False     # データにNaNが含まれている場合も何もしない
        else:
            input_data = tmp.values.reshape(-1)
            return self.model.predict([input_data])
            
    
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
            self.console.log(f"{self.disp_name}（{self.symbol}）：[cyan]データを更新しました。[/]")
            
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
                self.console.log(f"{self.disp_name}（{self.symbol}）：[blue]成行で売り注文を出しました（ロスカット）[/]\U0001F602")
                return True
            else:
                self.console.log(f"{self.disp_name}（{self.symbol}）：:warning:[red]条件により売り注文を出せませんでした。[/]")
                return False

        return False
            
        
    def check_and_update_buy_order_status(self):

        # 買い注文の約定状況を確認する
        result = self.lib.check_execution(self.buy_order_id)

        # 約定している場合
        if result['OrderState'] == 5:
            
            self.buy_order_flag = False
            self.purchase_price = result['Price']
            self.console.log(f"[yellow]{self.disp_name}（{self.symbol}）[/]を [red]{self.transaction_unit} 株 {self.purchase_price:,} 円で購入[/]しました \U0001F4B0")

            return True

        return False


    def check_and_update_sell_order_status(self):

        # 売り注文の約定状況を確認する
        result = self.lib.check_execution(self.sell_order_id)

        # 約定している場合
        if result['OrderState'] == 5:

            self.sell_order_flag = False
            self.loss_cut = False
            price = result['Price']
            pf = (price - self.purchase_price) * self.transaction_unit
            if pf >= 0:
                self.console.log(f"[yellow]{self.disp_name}（{self.symbol}）[/]を [red]{self.transaction_unit} 株 {price} 円で売却[/]し、利益が {pf:,} 円でした \U0001F60F")
            else:
                self.console.log(f"[yellow]{self.disp_name}（{self.symbol}）[/]を [red]{self.transaction_unit} 株 {price} 円で売却[/]し、損失が {pf:,} 円でした \U0001F622")
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
            self.console.log(f"{self.disp_name}（{self.symbol}）：[blue]指値で売り注文を出しました[/] \U0001F4B0")
            return True
        else:
            self.console.log(f"{self.disp_name}（{self.symbol}）：:warning:[red]売り注文を出せませんでした。[/]")
            return False
    
    
    def buy_at_market_price_with_cash(self):

        # まだ売り注文が残っている場合は買わない
        if self.sell_order_flag:
            self.console.log(f"{self.disp_name}（{self.symbol}）：[red]売り注文が残っているため、買い注文を出しませんでした。[/]")
            return False
        
        # 15:30まで20分を切っている場合は買わない
        now = datetime.now()
        target_time = datetime.combine(now.date(), time(15, 30))
        time_difference = target_time - now
        if time_difference <= timedelta(minutes = 20):
            self.console.log(f"{self.disp_name}（{self.symbol}）：[red]15:30まで20分を切っているので買い注文を出しませんでした。[/]")
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
                self.console.log(f"{self.disp_name}（{self.symbol}）：[blue]成行で買い注文を出しました[/] \U0001F4B0")
                return True
            else:
                self.console.log(f"{self.disp_name}（{self.symbol}）：:warning:[red]買い注文を出せませんでした。[/]")
                return False                            
            
        else:
            self.console.log(f"{self.disp_name}（{self.symbol}）：[red]値上がりが予測されましたが、買付余力がありませんでした。[/]")
            return False

    
