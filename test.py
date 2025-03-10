import time
import math
from playsound import playsound
from library import StockLibrary

from rich.console import Console

console = Console(log_time_format="%Y-%m-%d %H:%M:%S")


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

        self.buy_price = 0
        self.sell_price = 0

        # 預金残高（現物の買付余力）を問い合わせる
        deposit_before = self.lib.deposit()
        console.log(f"[yellow]買付余力 {int(deposit_before):,} 円[/]")

    def buy_stock(self):
        # 成行で買い注文を出す
        content = self.lib.buy_at_market_price_with_cash(self.symbol, self.unit)

        try:
            result = content["Result"]
        except KeyError:
            console.log(f"KeyError: {content}")
            return False

        if result == 0:
            buy_order_id = content["OrderId"]
            console.log("[blue]成行で買い注文を出しました[/]\U0001f4b8")

            # 買い注文の約定状況を確認する
            result = self.lib.check_execution(buy_order_id)

            try:
                state = result[0]["State"]
            except KeyError:
                console.log(f"KeyError: {result}")
                return False

            while state != 5:
                console.log("[blue]買い注文がまだ約定していません[/]\U0001f4b8")
                result = self.lib.check_execution(buy_order_id)
                state = result[0]["State"]
                time.sleep(1)

            self.buy_price = result[0]["Details"][2]["Price"]
            console.log(
                f"[blue]買い注文が {self.buy_price:,} 円で約定しました[/]\U0001f4b0"
            )
            playsound("./sound/buy.mp3")

            return True

        else:
            console.log("[red]買い注文が失敗しました[/]\U0001f6ab")
            console.log(content)

            return False

    def sell_stock(self):
        # 指値で売り注文を出す
        sell_price = math.ceil(self.buy_price * 1.005)
        content = self.lib.sell_at_limit_price(self.symbol, self.unit, sell_price)

        try:
            result = content["Result"]
        except KeyError:
            console.log(f"KeyError: {content}")
            return False

        if result == 0:
            sell_order_id = content["OrderId"]
            console.log(
                f"[blue]{sell_price:,} 円の指値で売り注文を出しました[/]\U0001f4b8"
            )

            # 売り注文の約定状況を確認する
            result = self.lib.check_execution(sell_order_id)

            try:
                state = result[0]["State"]
            except KeyError:
                console.log(f"KeyError: {result}")
                return False

            while state != 5:
                console.log("[blue]売り注文がまだ約定していません[/]\U0001f4b8")
                result = self.lib.check_execution(sell_order_id)
                state = result[0]["State"]
                time.sleep(1)

            self.sell_price = result[0]["Details"][2]["Price"]
            console.log(
                f"[blue]売り注文が {self.sell_price:,} 円で約定しました[/]\U0001f4b0"
            )

            return True

        else:
            console.log("[red]売り注文が失敗しました[/]\U0001f6ab")
            console.log(content)

            return False


if __name__ == "__main__":
    test = Test(
        [
            2552,
        ]
    )

    if test.buy_stock():
        if test.sell_stock():
            console.log("[green]取引が成功しました[/]\U0001f4b0")

            # 損益を計算する
            profit_loss = (test.sell_price - test.buy_price) * test.unit
            if profit_loss > 0:
                playsound("./sound/profit.mp3")
                console.log(f"[red]利益: {profit_loss} 円[/]")
            else:
                playsound("./sound/loss.mp3")
                console.log(f"[blue]損失: {profit_loss} 円[/]")
