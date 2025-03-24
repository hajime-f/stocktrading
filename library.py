import asyncio
import json
import os
import traceback
import urllib.request

import websockets
from dotenv import load_dotenv
from rich.console import Console

console = Console(log_time_format="%Y-%m-%d %H:%M:%S")


class StockLibrary:
    def __init__(self):
        # .envファイルから環境変数を読み込む
        load_dotenv()

        # APIパスワードの設定
        try:
            self.api_password = os.getenv("APIPassword_production")
        except KeyError:
            console.log(
                "\U000026a0[red]APIパスワードが環境変数に設定されていません。[/]"
            )
            exit()

        # IPアドレスの設定
        try:
            self.ip_address = os.getenv("IPAddress")
        except KeyError:
            console.log("\U000026a0[red]IPアドレスが環境変数に設定されていません。[/]")
            exit()

        # ポート番号の設定
        try:
            self.port = os.getenv("Port")
        except KeyError:
            console.log("\U000026a0[red]ポート番号が環境変数に設定されていません。[/]")
            exit()

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
            console.log("\U000026a0[red]APIトークンを取得できませんでした。[/]")
            exit()

        # Websocketの設定
        self.ws_uri = f"ws://{self.ip_address}:{self.port}/kabusapi/websocket"
        self.timeout_sec = 36000
        self.closed = asyncio.Event()

    def __call__(self, func):
        return func

    def register_receiver(self, func):
        # 受信関数を登録
        self.receive_func = func

    async def stream(self, func):
        while True:
            try:
                async with websockets.connect(
                    self.ws_uri, ping_timeout=self.timeout_sec
                ) as ws:
                    self.closed.clear()
                    while not self.closed.is_set():
                        try:
                            response = await asyncio.wait_for(
                                ws.recv(), timeout=self.timeout_sec
                            )
                            func(json.loads(response))
                        except (
                            websockets.exceptions.ConnectionClosedError,
                            websockets.exceptions.ConnectionClosedOK,
                        ) as e:
                            console.log(f"接続が閉じられました：{e}")
                            self.closed.set()
                            break
                        except asyncio.TimeoutError:
                            console.log("タイムアウトしました。")
                            self.closed.set()
                            break
                        except Exception as e:
                            console.log(f"エラーが発生しました：{e}")
                            traceback.print_exc()
                            self.closed.set()
                            break
            except Exception as e:
                console.log(f"接続エラーが発生しました：{e}")
                traceback.print_exc()
                self.closed.set()

    async def _run(self):
        await self.stream(self.receive_func)

    def run(self):
        self.loop = asyncio.get_event_loop()
        self.loop.run_until_complete(self._run())

        return True

    def register(self, symbol_list, exchange=1):
        # リストに含まれる銘柄を登録銘柄として登録する
        url = self.base_url + "/register"

        obj = {"Symbols": []}
        for symbol in symbol_list:
            obj["Symbols"].append({"Symbol": symbol, "Exchange": 1})

        content = self.put_request(url, obj)
        return content

    def unregister_all(self):
        # 登録銘柄リストからすべての銘柄を削除する
        url = self.base_url + "/unregister/all"
        req = urllib.request.Request(url, method="PUT")
        req.add_header("Content-Type", "application/json")
        req.add_header("X-API-KEY", self.token)
        content = self.throw_request(req)
        return content

    def deposit_cash(self):
        # 現物の取引余力を問い合わせる
        url = self.base_url + "/wallet/cash"
        content = self.get_request(url)
        return content["StockAccountWallet"]

    def deposit_margin(self):
        # 信用の取引余力を問い合わせる
        url = self.base_url + "/wallet/margin"
        content = self.get_request(url)
        return content["MarginAccountWallet"]

    def fetch_price(self, symbol, exchange):
        # ある銘柄の時価を得る
        content = self.fetch_board(symbol, exchange)
        return content["CurrentPrice"]

    def fetch_board(self, symbol, exchange):
        url = self.base_url + "/board/" + str(symbol) + "@" + str(exchange)
        content = self.get_request(url)
        return content

    def fetch_information(self, symbol, exchange):
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
            with urllib.request.urlopen(req) as res:
                content = json.loads(res.read())
        except urllib.error.HTTPError as e:
            console.log(f"\U000026a0[red]{e}[/]")
            content = json.loads(e.read())
        except Exception as e:
            console.log(f"\U000026a0[red]{e}[/]")

        return content

    def buy_at_market_price_with_cash(self, symbol, count, exchange=1):
        # 預かり金で成行買いする

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

    def execute_margin_buy_market_order_at_opening(self, symbol, count, exchange=1):
        # 寄付に信用で成行買いする（信用寄成）

        url = self.base_url + "/sendorder"

        obj = {
            "Symbol": symbol,  # 銘柄コード
            "Exchange": exchange,  # 市場
            "SecurityType": 1,  # 株式
            "Side": "2",  # 買い
            "CashMargin": 2,  # 新規
            "MarginTradeType": 1,  # 制度信用
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
        # 引けに信用で成行売りする（信用引成）

        url = self.base_url + "/sendorder"

        obj = {
            "Symbol": symbol,  # 銘柄コード
            "Exchange": exchange,  # 市場
            "SecurityType": 1,  # 株式
            "Side": "1",  # 売り
            "CashMargin": 3,  # 返済
            "MarginTradeType": 1,  # 制度信用
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
