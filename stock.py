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

        # 買いポジションを確認する
        buy_position = self.seek_position(side=2)

        if buy_position is None:
            # 買いポジションがない場合、寄付に信用で成行の買い注文を出す
            self.execute_margin_buy_market_order_at_opening()

        else:
            # 買いポジションがある場合、約定状況を確認する
            if len(buy_position) != 1:
                raise AssertionError("買いポジションが複数あります")

            if self.check_order_status(buy_position["Order_id"].values[0]):
                # 買い注文が約定している場合、引けに信用で成行の売り注文を出す
                self.execute_margin_sell_market_order_at_closing()

        # 売りポジションを確認する
        sell_position = self.seek_position(side=1)

        if sell_position is not None:
            # 売りポジションがある場合、約定状況を確認する
            if len(sell_position) != 1:
                raise AssertionError("売りポジションが複数あります")

            if self.check_order_status(sell_position["Order_id"].values[0]):
                console.log(f"{self.disp_name}（{self.symbol}）：[blue]売買成立[/]")

        # データを更新する
        self.update_data()
        self.time = []
        self.price = []
        self.volume = []

    def execute_margin_buy_market_order_at_opening(self):
        # 寄付に信用で成行の買い注文を入れる
        content = self.lib.execute_margin_buy_market_order_at_opening(
            self.symbol, self.transaction_unit, self.exchange
        )

        try:
            result = content["Result"]
        except KeyError:
            console.log(f"{self.disp_name}（{self.symbol}）：[red]発注失敗[/]")
            console.log(content)

        if result == 0:
            console.log(f"{self.disp_name}（{self.symbol}）：[blue]発注成功[/]")
            self.save_order(side=2, price=None, order_id=content["OrderId"])
        else:
            console.log(f"{self.disp_name}（{self.symbol}）：[red]発注失敗[/]")
            console.log(content)

    def check_order_status(self, order_id):
        # 注文の約定状況を確認する
        result = self.lib.check_orders(symbol=None, side=None, order_id=order_id)

        if not result:
            raise AssertionError(f"id：{order_id} に対応する約定情報が取得できません")

        # 約定している場合
        if result[0]["State"] == 5:
            price = int(result[0]["Details"][2]["Price"])
            console.log(
                f"[yellow]{self.disp_name}（{self.symbol}）[/]を [red]{self.price:,} 円で {self.transaction_unit} 株が約定[/]しました"
            )
            self.dm.update_price(order_id, price)

            return True

        return False

    def save_order(self, side, price, order_id):
        df_data = pd.DataFrame(
            {
                "DateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Symbol": self.symbol,
                "Displayname": self.disp_name,
                "Price": price,
                "Order_id": order_id,
                "Side": side,
            }
        )
        self.dm.save_order(df_data)
        self.buy_order_id = None
        self.purchase_price = None

    def seek_position(self, side):
        df_data = self.dm.load_order()

        today = datetime.now().strftime("%Y-%m-%d")
        df_data["DateTime"] = pd.to_datetime(df_data["DateTime"])

        df_extracted_data = df_data[
            (df_data["DateTime"].dt.date == pd.to_datetime(today).date())
            & (df_data["Symbol"] == self.symbol)
            & (df_data["Side"] == str(side))
        ]

        if df_extracted_data.empty:
            return None
        else:
            return df_extracted_data

    def execute_margin_sell_market_order_at_closing(self):
        content = self.lib.execute_margin_sell_market_order_at_closing(
            self.symbol, self.transaction_unit, self.exchange
        )

        try:
            result = content["Result"]
        except KeyError:
            console.log(f"{self.disp_name}（{self.symbol}）：[red]発注失敗[/]")
            console.log(content)

        if result == 0:
            console.log(f"{self.disp_name}（{self.symbol}）：[blue]発注成功[/]")
            self.save_order(side=1, price=None, order_id=content["OrderId"])
        else:
            console.log(f"{self.disp_name}（{self.symbol}）：[red]発注失敗[/]")
            console.log(content)
