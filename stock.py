import os
import sys
from datetime import datetime
from logging import getLogger

import pandas as pd
from dotenv import load_dotenv

from misc import MessageManager

logger = getLogger(__name__)
msg = MessageManager()


class Stock:
    def __init__(self, symbol, lib, dm, side, brand_name, exchange=1):
        self.symbol = symbol
        self.lib = lib
        self.dm = dm
        self.side = side
        self.brand_name = brand_name
        self.exchange = exchange

        load_dotenv()
        self.base_transaction = os.getenv("BaseTransaction")

        self.time = []
        self.price = []
        self.volume = []

        self.buy_executed = False
        self.sell_executed = False

        self.data = pd.DataFrame()

    def set_information(self):
        content = self.lib.fetch_information(self.symbol, self.exchange)
        try:
            self.disp_name = content["DisplayName"]
            self.unit = int(content["TradingUnit"])
            self.transaction_unit = self.unit * int(self.base_transaction)
        except KeyError as e:
            logger.critical(msg.get("errors.info_failed", symbol=self.symbol, reason=e))
            sys.exit(1)
        except Exception as e:
            logger.critical(msg.get("errors.info_failed", symbol=self.symbol, reason=e))
            sys.exit(1)

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

        if self.side == 1:
            self.sell_side()  # 売り注文
        elif self.side == 2:
            self.buy_side()  # 買い注文
        else:
            logger.warning(msg.get("errors.unexpected_side_value"))

        # データを更新する
        self.update_data()
        self.time, self.price, self.volume = [], [], []

    def sell_side(self):
        # 売り注文が完結していない場合、まずは売り注文（寄成）を約定させる
        if not self.sell_executed:
            # 売り注文の有無を確認する
            sell_position = self.dm.seek_position(symbol=self.symbol, side=1)

            if sell_position.empty:
                # まだ売り注文を入れていない場合、寄付での売り建てを試みる
                self.execute_margin_sell_market_order_at_opening()

            else:
                if len(sell_position) != 1:
                    logger.warning(
                        msg.get(
                            "errors.unexpected_orders_sell",
                            disp_name=self.disp_name,
                            symbol=self.symbol,
                        )
                    )

                # 注文IDを取得する
                order_id = sell_position["order_id"].values[0]

                # すでに売り注文を入れている場合、約定状況を確認する
                execution_data = self.check_order_status(order_id)
                if execution_data is not None:
                    # 売り注文が約定している（売り建てできている）場合、その情報を記録してフラグを立てる
                    self.record_execution(execution_data, order_id)
                    self.sell_executed = True

        # 売り注文は完結しているが、買い注文が完結していない場合、次に買い注文（引成）を約定させる
        if self.sell_executed and not self.buy_executed:
            # 買い注文の有無を確認する
            buy_position = self.dm.seek_position(symbol=self.symbol, side=2)

            if buy_position.empty:
                # まだ買い注文を入れていない場合、引けでの返済を試みる
                self.execute_margin_buy_market_order_at_closing()

            else:
                if len(buy_position) != 1:
                    logger.warning(
                        msg.get(
                            "errors.unexpected_orders_buy",
                            disp_name=self.disp_name,
                            symbol=self.symbol,
                        )
                    )

                # 注文IDを取得する
                order_id = buy_position["order_id"].values[0]

                # すでに買い注文を入れている場合、約定状況を確認する
                execution_data = self.check_order_status(order_id)
                if execution_data is not None:
                    # 買い注文が約定している（返済できている）場合、その情報を記録してフラグを立てる
                    self.record_execution(execution_data, order_id)
                    self.buy_executed = True

    def buy_side(self):
        # 買い注文が完結していない場合、まずは買い注文（寄成）を約定させる
        if not self.buy_executed:
            # 買い注文の有無を確認する
            buy_position = self.dm.seek_position(symbol=self.symbol, side=2)

            if buy_position.empty:
                # まだ買い注文を入れていない場合、寄付での買い建てを試みる
                self.execute_margin_buy_market_order_at_opening()

            else:
                if len(buy_position) != 1:
                    logger.warning(
                        msg.get(
                            "errors.unexpected_orders_buy",
                            disp_name=self.disp_name,
                            symbol=self.symbol,
                        )
                    )

                # 注文IDを取得する
                order_id = buy_position["order_id"].values[0]

                # すでに買い注文を入れている場合、約定状況を確認する
                execution_data = self.check_order_status(order_id)
                if execution_data is not None:
                    # 買い注文が約定している（買い建てできている）場合、その情報を記録してフラグを立てる
                    self.record_execution(execution_data, order_id)
                    self.buy_executed = True

        # 買い注文は完結しているが、売り注文が完結していない場合、次に売り注文（引成）を約定させる
        if self.buy_executed and not self.sell_executed:
            # 売り注文の有無を確認する
            sell_position = self.dm.seek_position(symbol=self.symbol, side=1)

            if sell_position.empty:
                # まだ売り注文を入れていない場合、引けでの返済を試みる
                self.execute_margin_sell_market_order_at_closing()

            else:
                if len(sell_position) != 1:
                    logger.warning(
                        msg.get(
                            "errors.unexpected_orders_sell",
                            disp_name=self.disp_name,
                            symbol=self.symbol,
                        )
                    )

                # 注文IDを取得する
                order_id = sell_position["order_id"].values[0]

                # すでに売り注文を入れている場合、約定状況を確認する
                execution_data = self.check_order_status(order_id)
                if execution_data is not None:
                    # 売り注文が約定している（返済できている）場合、その情報を記録してフラグを立てる
                    self.record_execution(execution_data, order_id)
                    self.sell_executed = True

    def execute_margin_buy_market_order_at_opening(self):
        # 寄付に信用で成行の買い注文を入れる（寄付買い建て）
        content = self.lib.execute_margin_buy_market_order_at_opening(
            self.symbol, self.transaction_unit, self.exchange
        )

        try:
            result = content["Result"]
        except KeyError:
            logger.error(
                msg.get(
                    "errors.buy_order_failed",
                    disp_name=self.disp_name,
                    symbol=self.symbol,
                )
            )
            logger.error(content)
            result = -1

        if result == 0:
            order_id = content["OrderId"]
            logger.info(
                msg.get(
                    "info.buy_order_success",
                    disp_name=self.disp_name,
                    symbol=self.symbol,
                    order_id=order_id,
                )
            )
            self.save_order(
                side=2,
                price=None,
                qty=self.transaction_unit,
                order_id=order_id,
            )
        else:
            logger.error(
                msg.get(
                    "errors.buy_order_failed",
                    disp_name=self.disp_name,
                    symbol=self.symbol,
                )
            )
            logger.error(content)

    def execute_margin_sell_market_order_at_opening(self):
        # 寄付に信用で成行の売り注文を入れる（寄付売り建て）
        content = self.lib.execute_margin_sell_market_order_at_opening(
            self.symbol, self.transaction_unit, self.exchange
        )

        try:
            result = content["Result"]
        except KeyError:
            logger.error(
                msg.get(
                    "errors.sell_order_failed",
                    disp_name=self.disp_name,
                    symbol=self.symbol,
                )
            )
            logger.error(content)
            result = -1

        if result == 0:
            order_id = content["OrderId"]
            logger.info(
                msg.get(
                    "info.sell_order_success",
                    disp_name=self.disp_name,
                    symbol=self.symbol,
                    order_id=order_id,
                )
            )
            self.save_order(
                side=1,
                price=None,
                qty=self.transaction_unit,
                order_id=order_id,
            )
        else:
            logger.error(
                msg.get(
                    "errors.sell_order_failed",
                    disp_name=self.disp_name,
                    symbol=self.symbol,
                )
            )
            logger.error(content)

    def check_order_status(self, order_id):
        # 注文の約定状況を確認する
        result = self.lib.check_orders(symbol=None, side=None, order_id=order_id)

        if not result:
            logger.error(msg.get("errors.execution_info_failed", order_id=order_id))
            return None

        if result[0]["State"] == 5:
            return result[0]
        else:
            return None

    def record_execution(self, data, order_id):
        # 約定情報がデータベースに保存されているかどうかを確認し、すでにある場合は何もしない
        df_execution = self.dm.load_execution(order_id)
        if not df_execution.empty:
            return

        side = int(data.get("Side"))
        recv_time = datetime.fromisoformat(data.get("RecvTime")).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # DetailsのRecTypeが8であるようなPriceとQtyを取得
        if "Details" in data and isinstance(data["Details"], list):
            for detail in data["Details"]:
                if detail.get("RecType") == 8:
                    price = detail.get("Price")
                    qty = detail.get("Qty")
                    ex_id = detail.get("ExecutionID")
                    ex_daytime = detail.get("ExecutionDay")
                    break
            ex_daytime = datetime.fromisoformat(ex_daytime).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        else:
            logger.error(msg.get("errors.execution_info_invalid"))
            logger.error(data)

        if float(price).is_integer():
            price = f"{int(price):,}"
        if float(qty).is_integer():
            qty = f"{int(qty):,}"

        if side == 1:
            logger.info(
                msg.get(
                    "info.sell_executed",
                    disp_name=self.disp_name,
                    symbol=self.symbol,
                    price=price,
                    qty=qty,
                )
            )
        elif side == 2:
            logger.info(
                msg.get(
                    "info.buy_executed",
                    disp_name=self.disp_name,
                    symbol=self.symbol,
                    price=price,
                    qty=qty,
                )
            )
        else:
            logger.warning(msg.get("errors.unexpected_side_value"))

        # 約定情報をデータベースに保存
        df_data = pd.DataFrame(
            {
                "exec_time": ex_daytime,
                "recv_time": recv_time,
                "symbol": self.symbol,
                "displayname": self.disp_name,
                "price": price,
                "qty": qty,
                "order_id": order_id,
                "execution_id": ex_id,
                "side": side,
            },
            index=[0],
        )
        self.dm.save_execution(df_data)

    def save_order(self, side, price, qty, order_id):
        df_data = pd.DataFrame(
            {
                "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": self.symbol,
                "displayname": self.disp_name,
                "price": price,
                "qty": qty,
                "order_id": order_id,
                "side": side,
            },
            index=[0],
        )
        self.dm.save_order(df_data)

    def execute_margin_sell_market_order_at_closing(self):
        # 引けに信用で成行の売り注文を入れる（引け返済）
        content = self.lib.execute_margin_sell_market_order_at_closing(
            self.symbol, self.transaction_unit, self.exchange
        )

        try:
            result = content["Result"]
        except KeyError:
            logger.error(
                msg.get(
                    "errors.sell_order_failed",
                    disp_name=self.disp_name,
                    symbol=self.symbol,
                )
            )
            logger.error(content)
            result = -1

        if result == 0:
            order_id = content["OrderId"]
            logger.info(
                msg.get(
                    "info.sell_order_success",
                    disp_name=self.disp_name,
                    symbol=self.symbol,
                    order_id=order_id,
                )
            )
            self.save_order(
                side=1,
                price=None,
                qty=self.transaction_unit,
                order_id=order_id,
            )
        else:
            logger.error(
                msg.get(
                    "errors.sell_order_failed",
                    disp_name=self.disp_name,
                    symbol=self.symbol,
                )
            )
            logger.error(content)

    def execute_margin_buy_market_order_at_closing(self):
        # 引けに信用で成行の買い注文を入れる（引け返済）
        content = self.lib.execute_margin_buy_market_order_at_closing(
            self.symbol, self.transaction_unit, self.exchange
        )

        try:
            result = content["Result"]
        except KeyError:
            logger.error(
                msg.get(
                    "errors.buy_order_failed",
                    disp_name=self.disp_name,
                    symbol=self.symbol,
                )
            )
            logger.error(content)
            result = -1

        if result == 0:
            order_id = content["OrderId"]
            logger.info(
                msg.get(
                    "info.buy_order_success",
                    disp_name=self.disp_name,
                    symbol=self.symbol,
                    order_id=order_id,
                )
            )
            self.save_order(
                side=2,
                price=None,
                qty=self.transaction_unit,
                order_id=order_id,
            )
        else:
            logger.error(
                msg.get(
                    "errors.buy_order_failed",
                    disp_name=self.disp_name,
                    symbol=self.symbol,
                )
            )
            logger.error(content)

    def check_transaction(self):
        if self.side == 1:
            if self.buy_executed and self.sell_executed:
                logger.info(
                    msg.get(
                        "info.sell_transaction_success",
                        symbol=self.symbol,
                        disp_name=self.disp_name,
                    )
                )
                return True
            elif self.buy_executed and not self.sell_executed:
                logger.warning(
                    msg.get(
                        "errors.transaction_failed_1",
                        symbol=self.symbol,
                        disp_name=self.disp_name,
                    )
                )
                return False
            else:
                logger.error(
                    msg.get(
                        "errors.transaction_failed_2",
                        symbol=self.symbol,
                        disp_name=self.disp_name,
                    )
                )
                return False

        elif self.side == 2:
            if self.buy_executed and self.sell_executed:
                logger.info(
                    msg.get(
                        "info.buy_transaction_success",
                        symbol=self.symbol,
                        disp_name=self.disp_name,
                    )
                )
                return True
            elif self.buy_executed and not self.sell_executed:
                logger.warning(
                    msg.get(
                        "errors.transaction_failed_3",
                        symbol=self.symbol,
                        disp_name=self.disp_name,
                    )
                )
                return False
            else:
                logger.error(
                    msg.get(
                        "errors.transaction_failed_4",
                        symbol=self.symbol,
                        disp_name=self.disp_name,
                    )
                )
                return False

        else:
            logger.warning(msg.get("errors.unexpected_side_value"))

    def fetch_prices(self):
        sell_position = self.dm.seek_execution(self.symbol, side=1)
        sell_price = (
            sell_position["price"].values[0] if not sell_position.empty else None
        )
        if sell_price is not None:
            sell_price = sell_price * sell_position["qty"].values[0]

        buy_position = self.dm.seek_execution(self.symbol, side=2)
        buy_price = buy_position["price"].values[0] if not buy_position.empty else None
        if buy_price is not None:
            buy_price = buy_price * buy_position["qty"].values[0]

        return sell_price, buy_price
