import os
import urllib.request
import json
import websockets
import asyncio
import traceback
from dotenv import load_dotenv


class StockLibrary:

    def __init__(self):

        # .envファイルから環境変数を読み込む
        load_dotenv()

        # APIパスワードの設定
        try:
            self.api_password = os.getenv("APIPassword_production")
        except KeyError:
            exit("APIパスワードが環境変数に設定されていません。")
        except Exception as e:
            print(e)
        
        # 取引パスワードの設定
        try:
            self.order_password = os.getenv("OrderPassword")
        except KeyError:
            exit("取引パスワードが環境変数に設定されていません。")
        except Exception as e:
            print(e)

        # IPアドレスの設定
        try:
            self.ip_address = os.getenv("IPAddress")
        except KeyError:
            exit("IPアドレスが環境変数に設定されていません。")
        except Exception as e:
            print(e)

        # ポート番号の設定
        try:
            self.port = os.getenv("Port")
        except KeyError:
            exit("ポート番号が環境変数に設定されていません。")
        except Exception as e:
            print(e)

        # エンドポイントの設定
        self.base_url = "http://" + self.ip_address + ":" +self.port + "/kabusapi/"

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
            exit("\033[31mAPIトークンを取得できませんでした。\033[0m")
        except Exception:
            exit("\033[31m不明な例外により強制終了します。\033[0m")

        # Websocketの設定
        self.ws_uri = "ws://" + self.ip_address + ":" + self.port + "/kabusapi/websocket"
        self.timeout_sec = 36000
        self.ping_interval = 180
        self.closed = asyncio.Event()


    def __call__(self, func):
        return func
        
    
    def register_receiver(self, func):

        # 受信関数を登録
        self.receive_func = func
        
        
    async def stream(func):
        
        while True:
            try:
                async with websockets.connect(self.ws_uri, ping_timeout=self.timeout_sec) as ws:
                    self.closed.clear()
                    while not self.closed.is_set():
                        try:
                            response = await asyncio.wait_for(ws.recv(), timeout=self.timeout_sec)
                            func(json.loads(response))
                        except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.ConnectionClosedOK) as e:
                            print(f"接続が閉じられました: {e}")
                            break
                        except asyncio.TimeoutError:
                            print("タイムアウトしました。再接続を試みます。")
                            break
                        except Exception as e:
                            print(f"エラーが発生しました: {e}")
                            traceback.print_exc()
                            break                        
            except Exception as e:
                print(f"接続エラーが発生しました: {e}")
                traceback.print_exc()
            finally:
                self.closed.set()

                
    async def _run(self):
        await self.stream(self.receive_func)

        
    def run(self):
        
        self.loop = asyncio.get_event_loop()
        self.loop.run_until_complete(self._run())

        return True        
                
        
    def register(self, symbol, exchange=1):

        # ある銘柄を登録銘柄リストに登録する
        url = self.base_url + '/register'
        obj = { 'Symbols':
                [ 
                    {'Symbol': str(symbol), 'Exchange': exchange},
                ] }
        content = self.put_request(url, obj)
        return content

    
    def unregister_all(self):

        # 登録銘柄リストからすべての銘柄を削除する
        url = self.base_url + '/unregister/all'
        req = urllib.request.Request(url, method='PUT')
        req.add_header('Content-Type', 'application/json')
        req.add_header('X-API-KEY', self.token)
        content = self.throw_request(req)
        return content


    def deposit(self):
        
        # 預金残高（現物の取引余力）を問い合わせる
        url = self.base_url + '/wallet/cash'
        content = self.get_request(url)
        return content['StockAccountWallet']
    

    def fetch_information(self, symbol, exchange):

        # ある銘柄の情報を得る
        url = self.base_url + '/symbol/' + str(symbol) + '@' + str(exchange)
        content = self.get_request(url)
        return content

    
    def put_request(self, url, obj):

        # PUT リクエストを url に送信する
        json_data = json.dumps(obj).encode('utf8')
        req = urllib.request.Request(url, json_data, method='PUT')
        req.add_header('Content-Type', 'application/json')
        req.add_header('X-API-KEY', self.token)

        content = self.throw_request(req)
        return content

    
    def throw_request(self, req):

        # リクエストを投げる
        try:
            with urllib.request.urlopen(req) as res:
                content = json.loads(res.read())
        except urllib.error.HTTPError as e:
            print('\033[31m'+ str(e) + '\033[0m')
            content = json.loads(e.read())
        except Exception as e:
            print('\033[31m' + str(e) + '\033[0m')

        return content


        
