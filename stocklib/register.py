import urllib.request
import json

class RegisterProcessor:

    def __init__(self, base_url, token):
        
        self.base_url = base_url
        self.token = token

    def push_unregisterall_request(self):

        # 登録銘柄リストからすべての銘柄を削除する
        url = self.base_url + '/unregister/all'
        req = urllib.request.Request(url, method='PUT')
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
        



