from datetime import datetime
import pandas as pd
from rich.console import Console

console = Console(log_time_format="%Y-%m-%d %H:%M:%S")


class Stock:
    def __init__(self, symbol, lib, dm, base_transaction, exchange=1):
        self.symbol = symbol
        self.lib = lib
        self.dm = dm
        self.base_transaction = base_transaction
        self.exchange = exchange

        self.time = []
        self.price = []
        self.volume = []

        self.buy_order_id = None
        self.purchase_price = None

        self.data = pd.DataFrame()

    def set_information(self):
        content = self.lib.fetch_information(self.symbol, self.exchange)
        try:
            self.disp_name = content["DisplayName"]
            self.unit = int(content["TradingUnit"])
            self.transaction_unit = self.unit * self.base_transaction
        except KeyError:
            console.log(f"[red] {self.symbol} の情報を取得できませんでした。[/]")
            exit()
        except Exception as e:
            console.log(f"[red]{self.symbol} の情報を取得できませんでした。[/]")
            console.log(f"[red]{e}[/]")
            exit()

    def append_data(self, new_data):
        if new_data["CurrentPriceTime"] is not None:
            dt_object = datetime.fromisoformat(
                new_data["CurrentPriceTime"].replace("Z", "+00:00")
            )
            self.time.append(dt_object.strftime("%Y-%m-%d %H:%M"))
            self.price.append(new_data["CurrentPrice"])
            self.volume.append(new_data["TradingVolume"])

    def update_data(self):
        if self.time:
            price_df = pd.DataFrame({"DateTime": self.time, "Price": self.price})
            price_df = price_df.set_index("DateTime")
            price_df.index = pd.to_datetime(price_df.index)
            price_df = price_df.resample("1Min").ohlc().dropna()
            price_df.columns = price_df.columns.get_level_values(1)

            self.data = pd.concat([self.data, price_df])

    def polling(self):
        """
        約５分間隔で呼ばれる関数
        """

        content = self.lib.fetch_positions(self.symbol, 2)
        console.log(content)
        content = self.lib.buy_at_market_price_with_margin(
            self.symbol, self.transaction_unit, self.exchange
        )
        console.log(content)
        content = self.lib.fetch_positions(self.symbol, 2)
        console.log(content)
        breakpoint()

        # 買いポジションを確認する
        if not self.lib.fetch_positions(self.symbol, 2):
            # 買いポジションがない場合、信用で成行の買い注文を出す
            self.buy_order_id = self.buy_at_market_price_with_margin()

        if self.buy_order_id is not None:
            # 買い注文が出せた場合、約定状況を確認する
            self.purchase_price = self.check_buy_order_status()

            if self.purchase_price is not None:
                # 約定している場合、データベースに保存する
                self.save_order(side=2)

        # データを更新する
        self.update_data()
        self.time = []
        self.price = []
        self.volume = []

    def buy_at_market_price_with_margin(self):
        # 信用で成行の買い注文を入れる
        content = self.lib.buy_at_market_price_with_margin(
            self.symbol, self.transaction_unit, self.exchange
        )

        try:
            result = content["Result"]
        except KeyError:
            console.log(f"{self.disp_name}（{self.symbol}）：[red]発注失敗[/]")
            console.log(content)
            return None

        if result == 0:
            console.log(f"{self.disp_name}（{self.symbol}）：[blue]成行発注成功[/]")
            return content["OrderId"]
        else:
            console.log(f"{self.disp_name}（{self.symbol}）：[red]発注失敗[/]")
            console.log(content)
            return None

    def check_buy_order_status(self):
        # 買い注文の約定状況を確認する
        result = self.lib.check_execution(self.buy_order_id)

        # 約定している場合
        if result[0]["State"] == 5:
            purchase_price = int(result[0]["Details"][2]["Price"])
            console.log(
                f"[yellow]{self.disp_name}（{self.symbol}）[/]を [red]{self.purchase_price:,} 円で {self.transaction_unit} 株購入[/]しました"
            )
            return purchase_price

        return None

    def save_order(self, side):
        df_data = pd.DataFrame(
            {
                "DateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Symbol": self.symbol,
                "Displayname": self.disp_name,
                "Price": self.price,
                "Order_id": self.buy_order_id,
                "Side": side,
            }
        )
        self.dm.save_order(df_data)
        self.buy_order_id = None
        self.purchase_price = None
