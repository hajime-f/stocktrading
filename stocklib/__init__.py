import os
import urllib.request
import json
from dotenv import load_dotenv

from stocklib import websocket
from stocklib import information

class Initialize:

    def __init__(self):

        # .envファイルから環境変数を読み込む
        # このファイルには、下記の要領でパスワードなどを記載しておく
        # 
        # APIPassword_production=XXXX
        # OrderPassword=YYYY
        # IPAddress=127.0.0.1（または localhost）
        # Port=:18080

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
        self.base_url = "http://" + self.ip_address + self.port + "/kabusapi/"

        # APIトークンの取得
        url = self.base_url + "/token"
        obj = {"APIPassword": self.api_password}
        json_data = json.dumps(obj).encode("utf8")
        
        req = urllib.request.Request(url, json_data, method="POST")
        req.add_header("Content-Type", "application/json")

        # リクエストを投げる
        try:
            with urllib.request.urlopen(req) as res:
                content = json.loads(res.read())
        except urllib.error.HTTPError as e:
            print("\033[31m"+ str(e) + "\033[0m")
            content = json.loads(e.read())
        except Exception as e:
            print("\033[31m" + str(e) + "\033[0m")

        try:
            self.token = content["Token"]
        except KeyError:
            exit("\033[31mAPIトークンを取得できませんでした。\033[0m")
        except Exception:
            exit("\033[31m不明な例外により強制終了します。\033[0m")

        # インスタンスを生成
        self.websocket = websocket.WebSocketStreamProcessor(self.ip_address, self.port)
        self.information = information.InformationProcessor(self.base_url, self.token)

        
