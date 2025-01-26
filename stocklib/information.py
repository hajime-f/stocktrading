import urllib.request
import json

class InformationProcessor:

    def __init__(self, base_url, token):
        
        self.base_url = base_url
        self.token = token

    def deposit(self):
        
        # 預金残高（現物の取引余力）を問い合わせる
        url = self.base_url + '/wallet/cash'
        content = self.get_request(url)
        return content['StockAccountWallet']
    
    def get_request(self, url):
        
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
            print('\033[31m'+ str(e) + '\033[0m')
            content = json.loads(e.read())
        except Exception as e:
            print('\033[31m' + str(e) + '\033[0m')

        return content
    
