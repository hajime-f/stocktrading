import os
import urllib.request
import json
import pprint
from dotenv import load_dotenv

class Deal:

    kabu_station_ip = '192.168.0.101'
    
    def __init__(self):

        # kabuステーションサーバのURL
        self.base_url = 'http://' + self.kabu_station_ip + ':8080/kabusapi/'

        # 環境変数の読み込み
        load_dotenv()
        
        # APIパスワードの設定
        try:
            self.api_password = os.getenv('APIPassword_production')
        except KeyError as e:
            print('API パスワードが環境変数に設定されていません。')
        except Exception as e:
            print(e)

        # 取引パスワードの設定
        try:
            self.order_password = os.getenv('OrderPassword')
        except KeyError as e:
            print('注文パスワードが環境変数に設定されていません。')
        except Exception as e:
            print(e)
        
        # APIトークンの取得
        url = self.base_url + '/token'
        obj = {'APIPassword': self.api_password}
        json_data = json.dumps(obj).encode('utf8')
        
        req = urllib.request.Request(url, json_data, method='POST')
        req.add_header('Content-Type', 'application/json')

        content = self.throw_request(req)
        self.token = content['Token']
                
        
    def inquiry_deposit(self):

        url = self.base_url + '/wallet/cash'
        content = self.push_get_request(url)
        return content['StockAccountWallet']


    def push_get_request(self, url):
        
        # GETリクエストをurlに送信する
        req = urllib.request.Request(url, method='GET')
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
            print(e)
            content = json.loads(e.read())
            print(content)
        except Exception as e:
            print(e)

        return content
    
