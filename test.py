import time
from playsound import playsound
from library import StockLibrary

from rich.console import Console
console = Console(log_time_format = "%Y-%m-%d %H:%M:%S")

class Test:

    def __init__(self, symbol_list, exchange=1):

        # 株ライブラリを初期化する
        self.lib = StockLibrary()

        # 登録銘柄リストからすべての銘柄を削除する
        self.lib.unregister_all()

        # 銘柄登録
        self.lib.register(symbol_list)

        # 銘柄情報を取得する
        content = self.lib.fetch_information(symbol_list[0], exchange)
        
        self.disp_name = content["DisplayName"]
        self.unit = int(content["TradingUnit"])
        
        self.symbol = symbol_list[0]
        self.exchange = exchange
        
        self.buy_price= 0
        self.sell_price = 0

        # 預金残高（現物の買付余力）を問い合わせる
        deposit_before = self.lib.deposit()
        console.log(f"[yellow]買付余力 {int(deposit_before):,} 円[/]")


    def buy_stock(self):

        # 成行で買い注文を出す
        content = self.lib.buy_at_market_price_with_cash(self.symbol, self.unit)

        try:
            result = content['Result']
        except KeyError:
            console.log(f"KeyError: {content}")
            return False

        if result == 0:
            
            buy_order_id = content['OrderId']
            console.log(f"[blue]成行で買い注文を出しました[/]\U0001F4B8")
        
            # 買い注文の約定状況を確認する
            result = self.lib.check_execution(buy_order_id)

            try:
                state = result[0]['State']
            except KeyError:
                console.log(f"KeyError: {result}")
                return False
            
            while state != 5:
                console.log(f"[blue]買い注文がまだ約定していません[/]\U0001F4B8")
                result = self.lib.check_execution(buy_order_id)
                state = result[0]['State']
                time.sleep(1)
                
            console.log(f"[blue]買い注文が約定しました[/]\U0001F4B0")
            playsound('./sound/buy.mp3')
            self.buy_price = result[0]['Price']

            return True

        else:

            console.log(f"[red]買い注文が失敗しました[/]\U0001F6AB")
            console.log(content)

            return False
        

    def sell_stock(self):

        # 成行で売り注文を出す
        content = self.lib.sell_at_market_price(self.symbol, self.unit)

        try:
            result = content['Result']
        except KeyError:
            console.log(f"KeyError: {content}")
            return False

        if result == 0:

            sell_order_id = content['OrderId']
            console.log(f"[blue]成行で売り注文を出しました[/]\U0001F4B8")
            
            # 売り注文の約定状況を確認する
            result = self.lib.check_execution(sell_order_id)

            try:
                state = result[0]['State']
            except KeyError:
                console.log(f"KeyError: {result}")
                return False
            
            while state != 5:
                console.log(f"[blue]売り注文がまだ約定していません[/]\U0001F4B8")
                result = self.lib.check_execution(sell_order_id)
                state = result[0]['State']
                time.sleep(1)
                
            console.log(f"[blue]売り注文が約定しました[/]\U0001F4B0")
            self.sell_price = result[0]['Price']

            return True

        else:

            console.log(f"[red]売り注文が失敗しました[/]\U0001F6AB")
            console.log(content)

            return False


if __name__ == '__main__':
    
    test = Test([1475,])
    
    if test.buy_stock():
        if test.sell_stock():
            console.log(f"[green]取引が成功しました[/]\U0001F4B0")
    
            # 損益を計算する
            profit_loss = (test.sell_price - test.buy_price) * test.unit
            if profit_loss > 0:
                playsound('./sound/profit.mp3')
                console.log(f"[red]利益: {profit_loss} 円[/]")
            else:
                playsound('./sound/loss.mp3')
                console.log(f"[blue]損失: {profit_loss} 円[/]")

    
        
        
        
        
        
