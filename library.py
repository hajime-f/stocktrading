import asyncio
import json
import time
import traceback
import urllib.request
import urllib.error
from logging import getLogger

import websockets

from config_manager import cm
from data_manager import DataManager
from exception import ConfigurationError, APIError
from misc import MessageManager


class Library:
    def __init__(self):
        self.logger = getLogger(f"{__name__}.library")
        self.msg = MessageManager()

        # APIパスワードの設定
        self.api_password = cm.get("api.kabu_station.api_password")
        if not self.api_password:
            self.logger.critical(self.msg.get("errors.api_not_found"))
            raise ConfigurationError

        # IPアドレスの設定
        self.ip_address = cm.get("api.kabu_station.ip_address")
        if not self.ip_address:
            self.logger.critical(self.msg.get("errors.ip_address_not_found"))
            raise ConfigurationError

        # ポート番号の設定
        self.port = cm.get("api.kabu_station.port")
        if not self.port:
            self.logger.critical(self.msg.get("errors.port_not_found"))
            raise ConfigurationError

        # エンドポイントの設定
        self.base_url = f"http://{self.ip_address}:{self.port}/kabusapi/"

        # APIトークンの取得
        url = self.base_url + "/token"
        obj = {"APIPassword": self.api_password}
        json_data = json.dumps(obj).encode("utf8")

        req = urllib.request.Request(url, json_data, method="POST")
        req.add_header("Content-Type", "application/json")
        content = self.throw_request(req)

        try:
            self.token = content["Token"]
        except KeyError:
            self.logger.critical(self.msg.get("errors.api_token_key_error"))
            raise APIError

        # Websocketの設定
        self.ws_uri = f"ws://{self.ip_address}:{self.port}/kabusapi/websocket"
        self.timeout_sec = int(cm.get("api.kabu_station.timeout_sec"))
        self.closed = asyncio.Event()

    def register_receiver(self, func):
        # 受信関数を登録
        self.receive_func = func

    async def stream(self, func, stop_event):
        while not stop_event.is_set():
            try:
                async with websockets.connect(
                    self.ws_uri, ping_timeout=self.timeout_sec
                ) as ws:
                    self.closed.clear()
                    while not self.closed.is_set() and not stop_event.is_set():
                        try:
                            response = await asyncio.wait_for(
                                ws.recv(), timeout=self.timeout_sec
                            )
                            func(json.loads(response))
                        except (
                            websockets.exceptions.ConnectionClosedError,
                            websockets.exceptions.ConnectionClosedOK,
                        ) as e:
                            self.logger.error(
                                self.msg.get("errors.connection_closed", reason=e)
                            )
                            self.closed.set()
                            break
                        except asyncio.TimeoutError:
                            self.logger.error(self.msg.get("errors.connection_timeout"))
                            self.closed.set()
                            break
                        except Exception as e:
                            self.logger.error(
                                self.msg.get("errors.connection_error", reason=e)
                            )
                            traceback.print_exc()
                            self.closed.set()
                            break
            except Exception as e:
                self.logger.error(self.msg.get("errors.connection_error", reason=e))
                traceback.print_exc()
                self.closed.set()

            if stop_event.is_set():
                break
            await asyncio.sleep(5)

    async def _run(self, stop_event):
        await self.stream(self.receive_func, stop_event)

    def run(self, stop_event):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.loop = loop

        try:
            self.logger.info(self.msg.get("info.push_thread_start"))
            self.loop.run_until_complete(self._run(stop_event))

        except Exception as e:
            self.logger.critical(
                self.msg.get("errors.push_thread_error", reason=e), exc_info=True
            )
            stop_event.set()

        finally:
            self.logger.info(self.msg.get("info.push_thread_end"))
            self.loop.close()

        return True

    def register(self, symbol_list, exchange=1):
        # リストに含まれる銘柄を登録銘柄として登録する
        url = self.base_url + "/register"

        obj = {"Symbols": []}
        for symbol in symbol_list:
            obj["Symbols"].append({"Symbol": symbol, "Exchange": 1})
        content = self.put_request(url, obj)

        try:
            result = content["RegistList"]
            if result:
                return
            else:
                self.logger.critical(self.msg.get("errors.register_failed"))
                raise APIError
        except KeyError:
            self.logger.critical(self.msg.get("errors.register_failed"))
            raise APIError

        return content

    def unregister_all(self):
        # 登録銘柄リストからすべての銘柄を削除する
        url = self.base_url + "/unregister/all"
        req = urllib.request.Request(url, method="PUT")
        req.add_header("Content-Type", "application/json")
        req.add_header("X-API-KEY", self.token)
        content = self.throw_request(req)

        try:
            result = content["RegistList"]
            if not result:
                return
            else:
                self.logger.critical(self.msg.get("errors.unregister_failed"))
                raise APIError
        except KeyError:
            self.logger.critical(self.msg.get("errors.unregister_failed"))
            raise APIError

    def wallet_cash(self):
        # 現物の取引余力を問い合わせる
        url = self.base_url + "/wallet/cash"
        content = self.get_request(url)

        try:
            result = content["StockAccountWallet"]
            if result:
                return result
            else:
                self.logger.critical(self.msg.get("errors.wallet_cash_not_found"))
                raise APIError
        except KeyError:
            self.logger.critical(self.msg.get("errors.wallet_cash_not_found"))
            raise APIError

    def wallet_margin(self):
        # 信用の取引余力を問い合わせる
        url = self.base_url + "/wallet/margin"
        content = self.get_request(url)

        try:
            result = content["MarginAccountWallet"]
            if result:
                return result
            else:
                self.logger.critical(self.msg.get("errors.wallet_cash_not_found"))
                raise APIError
        except KeyError:
            self.logger.critical(self.msg.get("errors.wallet_cash_not_found"))
            raise APIError

    def fetch_price(self, symbol, exchange=1):
        # ある銘柄の時価を得る
        content = self.fetch_board(symbol, exchange)
        return content["CurrentPrice"]

    def fetch_board(self, symbol, exchange=1):
        url = self.base_url + "/board/" + str(symbol) + "@" + str(exchange)
        content = self.get_request(url)
        return content

    def fetch_information(self, symbol, exchange=1):
        # ある銘柄の情報を得る
        url = self.base_url + "/symbol/" + str(symbol) + "@" + str(exchange)
        content = self.get_request(url)
        return content

    def put_request(self, url, obj):
        # PUT リクエストを url に送信する
        json_data = json.dumps(obj).encode("utf8")
        req = urllib.request.Request(url, json_data, method="PUT")
        req.add_header("Content-Type", "application/json")
        req.add_header("X-API-KEY", self.token)

        content = self.throw_request(req)
        return content

    def get_request(self, url, obj=None):
        # GETリクエストをurlに送信する
        if obj is None:
            req = urllib.request.Request(url, method="GET")
        else:
            req = urllib.request.Request(
                "{}?{}".format(url, urllib.parse.urlencode(obj)), method="GET"
            )
        req.add_header("Content-Type", "application/json")
        req.add_header("X-API-KEY", self.token)

        content = self.throw_request(req)
        return content

    def post_request(self, url, obj):
        # POST リクエストを url に送信する

        json_data = json.dumps(obj).encode("utf-8")
        req = urllib.request.Request(url, json_data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("X-API-KEY", self.token)

        content = self.throw_request(req)
        return content

    def throw_request(self, req):
        # リクエストを投げる
        try:
            with urllib.request.urlopen(req, timeout=10) as res:
                return json.loads(res.read())
        except urllib.error.HTTPError as e:
            self.logger.critical(
                self.msg.get("errors.http_error", code=e.code, reason=e.reason)
            )
            try:
                error_content = json.loads(e.read())
                self.logger.error(f"APIエラーレスポンス: {error_content}")
                return error_content
            except json.JSONDecodeError:
                self.logger.error(self.msg.get("errors.decode_error"))
                raise APIError(
                    f"HTTP Error {e.code}, but failed to parse error body."
                ) from e
        except urllib.error.URLError as e:
            self.logger.critical(self.msg.get("errors.url_error", reason=e.reason))
            raise APIError("Network connection failed.") from e
        except json.JSONDecodeError as e:
            self.logger.critical(self.msg.get("errors.decode_error"))
            raise APIError("Invalid JSON response from API.") from e
        except Exception as e:
            self.logger.critical(self.msg.get("errors.http_other_error", reason=e))

    def buy_at_market_price_with_cash(self, symbol, count, exchange=1):
        # 預かり金で成行買いする→現物の成行買い

        url = self.base_url + "/sendorder"

        obj = {
            "Symbol": symbol,  # 銘柄コード
            "Exchange": exchange,  # 市場
            "SecurityType": 1,  # 株式
            "Side": "2",  # 買い
            "CashMargin": 1,  # 現物
            "DelivType": 2,  # 預り金
            "FundType": "AA",  # 信用代用
            "AccountType": 4,  # 特定口座
            "Qty": count,  # 注文数量
            "FrontOrderType": 10,  # 執行条件（成行）
            "Price": 0,  # 注文価格（成行なのでゼロ）
            "ExpireDay": 0,  # 当日中
        }
        content = self.post_request(url, obj)

        return content

    def sell_at_limit_price(self, symbol, count, limit_price, exchange=1):
        # 現物を指値で売る
        url = self.base_url + "/sendorder"

        obj = {
            "Symbol": symbol,  # 銘柄コード
            "Exchange": exchange,  # 市場
            "SecurityType": 1,  # 株式
            "Side": "1",  # 売り
            "CashMargin": 1,  # 現物
            "DelivType": 0,  # 預かり金
            "FundType": "  ",  # 現物売
            "AccountType": 4,  # 特定口座
            "Qty": count,  # 注文数量
            "FrontOrderType": 20,  # 執行条件（指値）
            "Price": limit_price,  # 指値
            "ExpireDay": 0,  # 当日中
        }

        content = self.post_request(url, obj)

        return content

    def sell_at_market_price(self, symbol, count, exchange=1):
        # 現物を成行で売る（ロスカット）
        url = self.base_url + "/sendorder"

        obj = {
            "Symbol": symbol,  # 銘柄コード
            "Exchange": exchange,  # 市場
            "SecurityType": 1,  # 株式
            "Side": "1",  # 売り
            "CashMargin": 1,  # 現物
            "DelivType": 0,  # 預かり金
            "FundType": "  ",  # 現物売
            "AccountType": 4,  # 特定口座
            "Qty": count,  # 注文数量
            "FrontOrderType": 10,  # 執行条件（成行）
            "Price": 0,  # 注文価格（成行なのでゼロ）
            "ExpireDay": 0,  # 当日中
        }

        content = self.post_request(url, obj)

        return content

    def execute_margin_buy_market_order_at_opening(self, symbol, count, exchange=1):
        # 寄付に信用で成行買いする（信用寄成）→寄り付きに買い建てる（新規の買い建玉）

        url = self.base_url + "/sendorder"

        obj = {
            "Symbol": symbol,  # 銘柄コード
            "Exchange": exchange,  # 市場
            "SecurityType": 1,  # 株式
            "Side": "2",  # 買い
            "CashMargin": 2,  # 新規
            "MarginTradeType": 3,  # 一般信用（デイトレ）
            "DelivType": 0,  # 指定なし
            "FundType": "11",  # 信用取引
            "AccountType": 4,  # 特定口座
            "Qty": count,  # 注文数量
            "FrontOrderType": 13,  # 執行条件（寄成）
            "Price": 0,  # 注文価格（成行なのでゼロ）
            "ExpireDay": 0,  # 当日中
        }
        content = self.post_request(url, obj)

        return content

    def execute_margin_sell_market_order_at_opening(self, symbol, count, exchange=1):
        # 寄付に信用で成行売りする（信用寄成）→寄り付きに売り建てる（新規の売り建玉）

        url = self.base_url + "/sendorder"

        obj = {
            "Symbol": symbol,  # 銘柄コード
            "Exchange": exchange,  # 市場
            "SecurityType": 1,  # 株式
            "Side": "1",  # 売り
            "CashMargin": 2,  # 新規
            "MarginTradeType": 3,  # 一般信用（デイトレ）
            "DelivType": 0,  # 指定なし
            "FundType": "11",  # 信用取引
            "AccountType": 4,  # 特定口座
            "Qty": count,  # 注文数量
            "FrontOrderType": 13,  # 執行条件（寄成）
            "Price": 0,  # 注文価格（成行なのでゼロ）
            "ExpireDay": 0,  # 当日中
        }
        content = self.post_request(url, obj)

        return content

    def execute_margin_sell_market_order_at_closing(self, symbol, count, exchange=1):
        # 引けに信用で成行売りする（信用引成）→引けに売り建てる（買い建玉の返済）

        url = self.base_url + "/sendorder"

        obj = {
            "Symbol": symbol,  # 銘柄コード
            "Exchange": exchange,  # 市場
            "SecurityType": 1,  # 株式
            "Side": "1",  # 売り
            "CashMargin": 3,  # 返済
            "MarginTradeType": 3,  # 一般信用（デイトレ）
            "DelivType": 2,  # 預かり金
            "FundType": "11",  # 信用取引
            "AccountType": 4,  # 特定口座
            "Qty": count,  # 注文数量
            "ClosePositionOrder": 1,  # 決済順序
            "FrontOrderType": 16,  # 執行条件（引成）
            "Price": 0,  # 注文価格（成行なのでゼロ）
            "ExpireDay": 0,  # 当日中
        }
        content = self.post_request(url, obj)

        return content

    def execute_margin_buy_market_order_at_closing(self, symbol, count, exchange=1):
        # 引けに信用で成行買いする（信用引成）→引けに買い建てる（売り建玉の返済）

        url = self.base_url + "/sendorder"

        obj = {
            "Symbol": symbol,  # 銘柄コード
            "Exchange": exchange,  # 市場
            "SecurityType": 1,  # 株式
            "Side": "2",  # 買い
            "CashMargin": 3,  # 返済
            "MarginTradeType": 3,  # 一般信用（デイトレ）
            "DelivType": 2,  # 預かり金
            "FundType": "11",  # 信用取引
            "AccountType": 4,  # 特定口座
            "Qty": count,  # 注文数量
            "ClosePositionOrder": 1,  # 決済順序
            "FrontOrderType": 16,  # 執行条件（引成）
            "Price": 0,  # 注文価格（成行なのでゼロ）
            "ExpireDay": 0,  # 当日中
        }
        content = self.post_request(url, obj)

        return content

    def execute_margin_sell_market_order_at_market(self, symbol, count, exchange=1):
        # 信用で成行売りする（ロスカット）→売り建てる（買い建玉の返済）

        url = self.base_url + "/sendorder"

        obj = {
            "Symbol": symbol,  # 銘柄コード
            "Exchange": exchange,  # 市場
            "SecurityType": 1,  # 株式
            "Side": "1",  # 売り
            "CashMargin": 3,  # 返済
            "MarginTradeType": 3,  # 一般信用（デイトレ）
            "DelivType": 2,  # 預かり金
            "FundType": "11",  # 信用取引
            "AccountType": 4,  # 特定口座
            "Qty": count,  # 注文数量
            "ClosePositionOrder": 1,  # 決済順序
            "FrontOrderType": 10,  # 執行条件（引成）
            "Price": 0,  # 注文価格（成行なのでゼロ）
            "ExpireDay": 0,  # 当日中
        }
        content = self.post_request(url, obj)

        return content

    def execute_margin_buy_market_order_at_market(self, symbol, count, exchange=1):
        # 信用で成行買いする（ロスカット）→買い建てる（売り建玉の返済）

        url = self.base_url + "/sendorder"

        obj = {
            "Symbol": symbol,  # 銘柄コード
            "Exchange": exchange,  # 市場
            "SecurityType": 1,  # 株式
            "Side": "2",  # 買い
            "CashMargin": 3,  # 返済
            "MarginTradeType": 3,  # 一般信用（デイトレ）
            "DelivType": 2,  # 預かり金
            "FundType": "11",  # 信用取引
            "AccountType": 4,  # 特定口座
            "Qty": count,  # 注文数量
            "ClosePositionOrder": 1,  # 決済順序
            "FrontOrderType": 10,  # 執行条件（成行）
            "Price": 0,  # 注文価格（成行なのでゼロ）
            "ExpireDay": 0,  # 当日中
        }
        content = self.post_request(url, obj)

        return content

    def check_orders(self, symbol, side, order_id=None):
        # 注文のステータスを確認する
        url = self.base_url + "/orders"

        if order_id is not None:
            obj = {
                "id": order_id,
            }
        else:
            obj = {
                "symbol": symbol,
                "side": side,
            }
        content = self.get_request(url, obj)

        return content

    def cancel_order(self, order_id):
        # 注文をキャンセルする
        url = self.base_url + "/cancelorder"

        obj = {
            "OrderId": order_id,
        }
        content = self.put_request(url, obj)

        return content

    def fetch_positions(self, symbol, side):
        # 保有ポジションを取得する
        url = self.base_url + "/positions"

        obj = {
            "product": 0,
            "symbol": symbol,
            "side": str(side),
            "addinfo": False,
        }
        content = self.get_request(url, obj)

        return content

    def examine_regulation(self, symbol, exchange=1):
        url = self.base_url + "/regulations/" + str(symbol) + "@" + str(exchange)
        content = self.get_request(url)

        try:
            return True if content["RegulationsInfo"] else False
        except KeyError:
            print(symbol)
            return True

    def examine_premium(self, symbol, exchange=1):
        url = self.base_url + "/margin/marginpremium/" + str(symbol)
        content = self.get_request(url)

        try:
            return content["DayTrade"]["MarginPremium"]
        except KeyError:
            print(symbol)
            return True


if __name__ == "__main__":
    dm = DataManager()
    lib = Library()

    orders = dm.load_order()

    for _, row in orders.iterrows():
        order_id = row["order_id"]
        content = lib.cancel_order(order_id)
        dm.delete_order(order_id)
        time.sleep(0.3)
