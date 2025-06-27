from datetime import datetime
from logging import getLogger

import pandas as pd

from exception import DataProcessingError
from misc import MessageManager

SIDE_SELL = 1
SIDE_BUY = 2
STOP_LOSS_RATE = 0.05


class Stock:
    def __init__(self, symbol, lib, dm, side, brand_name, base_transaction, exchange=1):
        self.symbol = symbol
        self.lib = lib
        self.dm = dm
        self.side = side
        self.brand_name = brand_name
        self.base_transaction = base_transaction
        self.exchange = exchange

        self.time, self.price, self.volume = [], [], []
        self.buy_executed, self.sell_executed = False, False

        self.data = pd.DataFrame()

        self.entry_price = 0.0

        self.logger = getLogger(f"{__name__}.{self.symbol}")
        self.msg = MessageManager()

        self.state = EntryOrderState(self)

    def set_information(self):
        try:
            content = self.lib.fetch_information(self.symbol, self.exchange)
            self.disp_name = content["DisplayName"]
            self.unit = int(content["TradingUnit"])
            self.transaction_unit = self.unit * int(self.base_transaction)
        except (KeyError, TypeError, ValueError, Exception) as e:
            raise DataProcessingError(
                self.msg.get("errors.info_failed", symbol=self.symbol)
            ) from e

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

    def set_state(self, new_state):
        # 状態を切り替えるメソッド
        self.state = new_state

    def polling(self):
        """
        約５分間隔で呼ばれる関数
        """

        # データを更新する
        self.update_data()
        self.time, self.price, self.volume = [], [], []

        # 実際の処理は状態オブジェクトに委譲する
        self.state.handle_polling()

    def handle_entry_order(self):
        # 新規建て注文の処理（発注または約定確認）を行う
        entry_side = SIDE_SELL if self.side == SIDE_SELL else SIDE_BUY

        df_position = self.dm.seek_position(symbol=self.symbol, side=entry_side)
        if df_position.empty:
            # まだ注文を入れていない場合、寄付での取引を試みる
            if entry_side == SIDE_SELL:
                self.execute_margin_sell_market_order_at_opening()
            else:
                self.execute_margin_buy_market_order_at_opening()
            return False
        else:
            # すでに注文を入れている場合、約定状況を確認する
            return self.check_execution(df_position, side=entry_side)

    def check_execution(self, df_position, side):
        # 注文の約定状況を確認し、状態フラグを更新する共通ロジック
        if len(df_position) != 1:
            error_key = (
                "errors.unexpected_orders_sell"
                if side == SIDE_SELL
                else "errors.unexpected_orders_buy"
            )
            self.logger.critical(
                self.msg.get(error_key, disp_name=self.disp_name, symbol=self.symbol)
            )
            raise DataProcessingError

        order_id = df_position["order_id"].values[0]
        execution_data = self.check_order_status(order_id)

        if execution_data:
            self.record_execution(execution_data, order_id)
            if side == SIDE_SELL:
                self.sell_executed = True
            else:
                self.buy_executed = True
            return True
        else:
            return False

    def handle_exit_order(self):
        # 返済注文の処理（発注または約定確認）を行う
        exit_side = SIDE_BUY if self.side == SIDE_SELL else SIDE_SELL

        df_position = self.dm.seek_position(symbol=self.symbol, side=exit_side)
        if df_position.empty:
            # まだ注文を入れていない場合：
            # 15:25分を過ぎている場合は引け取引の注文を入れる
            now = datetime.now()
            if now.hour > 15 or (now.hour == 15 and now.minute >= 25):
                if exit_side == SIDE_SELL:
                    self.execute_margin_sell_market_order_at_closing()
                else:
                    self.execute_margin_buy_market_order_at_closing()

            # 損切りの要否を確認し、必要であれば損切り注文を入れる
            elif self.check_stop_loss():
                if exit_side == SIDE_SELL:
                    self.execute_margin_sell_market_order_at_market()
                else:
                    self.execute_margin_buy_market_order_at_market()

            return False
        else:
            # すでに注文を入れている場合、約定状況を確認する
            return self.check_execution(df_position, side=exit_side)

    def check_stop_loss(self):
        current_price = self.data.tail(1)["close"].item()
        flag = False

        if self.side == SIDE_SELL:
            if current_price >= self.entry_price * (1 + STOP_LOSS_RATE):
                flag = True
        elif self.side == SIDE_BUY:
            if current_price <= self.entry_price * (1 - STOP_LOSS_RATE):
                flag = True

        if flag:
            self.logger.info(
                self.msg.get(
                    "info.stop_loss_triggered",
                    disp_name=self.disp_name,
                    symbol=self.symbol,
                    entry_price=f"{self.entry_price:,.0f}",
                    current_price=f"{current_price:,.0f}",
                )
            )
            return True
        else:
            return False

    def check_order_status(self, order_id):
        # 注文の約定状況を確認する
        result = self.lib.check_orders(symbol=None, side=None, order_id=order_id)

        if not result:
            self.logger.error(
                self.msg.get("errors.execution_info_failed", order_id=order_id)
            )
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
        price, qty, ex_id, ex_daytime = None, None, None, None
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
            self.logger.error(self.msg.get("errors.execution_info_invalid"))
            self.logger.error(data)
            return

        if price is None:
            self.logger.error(self.msg.get("errors.execution_info_invalid"))
            self.logger.error(data)
            return

        # 約定価格を保持しておく
        self.entry_price = float(price)

        msg_key = ""
        if side == SIDE_SELL:
            msg_key = "info.sell_executed"
        elif side == SIDE_BUY:
            msg_key = "info.buy_executed"

        if msg_key:
            self.logger.info(
                self.msg.get(
                    msg_key,
                    disp_name=self.disp_name,
                    symbol=self.symbol,
                    price=f"{int(price):,}" if float(price).is_integer() else price,
                    qty=f"{int(qty):,}" if float(qty).is_integer() else qty,
                )
            )
        else:
            self.logger.warning(self.msg.get("errors.unexpected_side_value"))

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

    def execute_margin_sell_market_order_at_market(self):
        # 損切り（ロスカット）のために信用で成行の売り注文を入れる
        content = self.lib.execute_margin_sell_market_order_at_market(
            self.symbol, self.transaction_unit, self.exchange
        )
        self.push_message_and_save_order(
            content, SIDE_SELL, None, self.transaction_unit
        )

    def execute_margin_buy_market_order_at_market(self):
        # 損切り（ロスカット）のために信用で成行の買い注文を入れる
        content = self.lib.execute_margin_buy_market_order_at_market(
            self.symbol, self.transaction_unit, self.exchange
        )
        self.push_message_and_save_order(content, SIDE_BUY, None, self.transaction_unit)

    def execute_margin_sell_market_order_at_opening(self):
        # 寄付に信用で成行の売り注文を入れる（寄付売り建て）
        content = self.lib.execute_margin_sell_market_order_at_opening(
            self.symbol, self.transaction_unit, self.exchange
        )
        self.push_message_and_save_order(
            content, SIDE_SELL, None, self.transaction_unit
        )

    def execute_margin_buy_market_order_at_opening(self):
        # 寄付に信用で成行の買い注文を入れる（寄付買い建て）
        content = self.lib.execute_margin_buy_market_order_at_opening(
            self.symbol, self.transaction_unit, self.exchange
        )
        self.push_message_and_save_order(content, SIDE_BUY, None, self.transaction_unit)

    def execute_margin_sell_market_order_at_closing(self):
        # 引けに信用で成行の売り注文を入れる（引け返済）
        content = self.lib.execute_margin_sell_market_order_at_closing(
            self.symbol, self.transaction_unit, self.exchange
        )
        self.push_message_and_save_order(
            content, SIDE_SELL, None, self.transaction_unit
        )

    def execute_margin_buy_market_order_at_closing(self):
        # 引けに信用で成行の買い注文を入れる（引け返済）
        content = self.lib.execute_margin_buy_market_order_at_closing(
            self.symbol, self.transaction_unit, self.exchange
        )
        self.push_message_and_save_order(content, SIDE_BUY, None, self.transaction_unit)

    def push_message_and_save_order(self, content, side, price, qty):
        msg_key_success = ""
        msg_key_failed = ""

        if side == SIDE_SELL:
            msg_key_success = "info.sell_order_success"
            msg_key_failed = "errors.sell_order_failed"
        elif side == SIDE_BUY:
            msg_key_success = "info.buy_order_success"
            msg_key_failed = "errors.buy_order_failed"

        try:
            result = content["Result"]
        except KeyError:
            self.logger.error(
                self.msg.get(
                    msg_key_failed,
                    disp_name=self.disp_name,
                    symbol=self.symbol,
                )
            )
            self.logger.error(content)
            result = -1

        if result == 0:
            order_id = content["OrderId"]
            self.logger.info(
                self.msg.get(
                    msg_key_success,
                    disp_name=self.disp_name,
                    symbol=self.symbol,
                    order_id=order_id,
                )
            )
            self.save_order(
                side=side,
                price=price,
                qty=qty,
                order_id=order_id,
            )
        else:
            self.logger.error(
                self.msg.get(
                    msg_key_failed,
                    disp_name=self.disp_name,
                    symbol=self.symbol,
                )
            )
            self.logger.error(content)

    def check_transaction(self):
        if self.side == SIDE_SELL:
            if self.buy_executed and self.sell_executed:
                self.logger.info(
                    self.msg.get(
                        "info.sell_transaction_success",
                        symbol=self.symbol,
                        disp_name=self.disp_name,
                    )
                )
                return True
            elif not self.buy_executed and self.sell_executed:
                self.logger.warning(
                    self.msg.get(
                        "errors.transaction_failed_1",
                        symbol=self.symbol,
                        disp_name=self.disp_name,
                    )
                )
                return False
            else:
                self.logger.error(
                    self.msg.get(
                        "errors.transaction_failed_2",
                        symbol=self.symbol,
                        disp_name=self.disp_name,
                    )
                )
                return False

        elif self.side == SIDE_BUY:
            if self.buy_executed and self.sell_executed:
                self.logger.info(
                    self.msg.get(
                        "info.buy_transaction_success",
                        symbol=self.symbol,
                        disp_name=self.disp_name,
                    )
                )
                return True
            elif self.buy_executed and not self.sell_executed:
                self.logger.warning(
                    self.msg.get(
                        "errors.transaction_failed_3",
                        symbol=self.symbol,
                        disp_name=self.disp_name,
                    )
                )
                return False
            else:
                self.logger.error(
                    self.msg.get(
                        "errors.transaction_failed_4",
                        symbol=self.symbol,
                        disp_name=self.disp_name,
                    )
                )
                return False

        else:
            self.logger.warning(self.msg.get("errors.unexpected_side_value"))

    def fetch_prices(self):
        sell_position = self.dm.seek_execution(self.symbol, side=SIDE_SELL)
        sell_price = None
        if not sell_position.empty:
            price = sell_position["price"].item()
            qty = sell_position["qty"].item()
            sell_price = price * qty

        buy_position = self.dm.seek_execution(self.symbol, side=SIDE_BUY)
        buy_price = None
        if not buy_position.empty:
            price = buy_position["price"].item()
            qty = buy_position["qty"].item()
            buy_price = price * qty

        return sell_price, buy_price


class TradingState:
    def __init__(self, stock):
        self.stock = stock

    def handle_polling(self):
        raise NotImplementedError


class EntryOrderState(TradingState):
    # 状態1: 新規建て注文を出す前の状態

    def handle_polling(self):
        if self.stock.handle_entry_order():
            self.stock.set_state(ExitOrderState(self.stock))


class ExitOrderState(TradingState):
    # 状態2: 返済注文を出す前の状態

    def handle_polling(self):
        if self.stock.handle_exit_order():
            self.stock.set_state(TradeCompleteState(self.stock))

        # 将来的にリアルタイム処理を入れる予定


class TradeCompleteState(TradingState):
    # 状態3: 全ての取引が完了した状態

    def handle_polling(self):
        pass
