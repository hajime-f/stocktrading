import os
import urllib.request
import json
import websockets
import asyncio
import traceback
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# from sktime.utils.plotting import plot_series
# from sktime.forecasting.ets import AutoETS
from sktime.forecasting.arima import AutoARIMA

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
        self.timeout_sec = 3600
        self.ping_interval = 60
        self.closed = asyncio.Event()

    
    def __call__(self, func):

        async def stream(func):
            
            try:
                async with websockets.connect(self.ws_uri, ping_timeout=self.timeout_sec) as ws:

                    # 最後にメッセージを受信した時刻を記録
                    last_message_time = asyncio.get_event_loop().time()
                    last_ping_time = last_message_time

                    while True:
                        try:
                            if asyncio.get_event_loop().time() - last_message_time > self.timeout_sec:
                                print("タイムアウトしました。接続を閉じます。")
                                await ws.close(code=1011, reason="Timeout")
                                break

                            if asyncio.get_event_loop().time() - last_ping_time > self.ping_interval:
                                await ws.ping()
                                last_ping_time = asyncio.get_event_loop().time()
                            
                            response = await asyncio.wait_for(ws.recv(), timeout=self.timeout_sec)
                            func(json.loads(response))

                            # 受信時刻を更新
                            last_message_time = asyncio.get_event_loop().time()  

                        except asyncio.TimeoutError:
                            print("タイムアウトしました。接続を閉じます。")
                            await ws.close()
                            break

                        except websockets.exceptions.ConnectionClosedError as e:
                            print(f"接続が閉じられました: {e}")
                            break
                    
                        except websockets.exceptions.ConnectionClosedOK:
                            print("サーバーから切断されました。")
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
            
        self.loop = asyncio.get_event_loop()
        self.loop.create_task(stream(func))
        return stream


    def run(self):

        async def wait_and_train():
            await self.closed.wait()  # 接続が閉じるまで待つ
            print("学習を開始します。")
            self.prepare_training_data()
            self.train_model()
            print("学習が完了しました。")

        self.loop.create_task(wait_and_train())
        self.loop.run_forever()


    def initialize_data(self, n_symbols):
        
        self.n_symbols = n_symbols
        
        self.data = []
        self.high_price = []
        self.low_price = []
        
        for _ in range(n_symbols):
            self.data.append([])
            self.high_price.append(0)
            self.low_price.append(1000000)


    def append_data(self, new_data, index):

        self.data[index].append(new_data)

        if self.high_price[index] < new_data['HighPrice']:
            self.high_price[index] = new_data['HighPrice']
        if self.low_price[index] > new_data['LowPrice']:
            self.low_price[index] = new_data['LowPrice']
    
        
    def prepare_training_data(self):

        columns = ['DateTime', 'Price']
        self.training_data = []

        df_data = []
        for i in range(self.n_symbols):
            df_data.append([])
            max_val = self.high_price[i]
            min_val = self.low_price[i]
            for d in self.data[i]:
                try:
                    dt_object = datetime.fromisoformat(d['CurrentPriceTime'].replace('Z', '+00:00'))
                    formatted_datetime = dt_object.strftime("%Y-%m-%d %H:%M")
                except ValueError:
                    print("文字列のフォーマットが異なります。")
                    exit()

                # 数値が-1〜1に収まるように正規化する
                price = (2 * (d['CurrentPrice'] - min_val) / (max_val - min_val)) - 1
                df_data[i].append([formatted_datetime, price])
        
        for _ in range(self.n_symbols):
            self.training_data.append(pd.DataFrame(index = [], columns = columns))
            
        for i in range(self.n_symbols):
            df = pd.DataFrame(df_data[i], columns = columns)
            if self.training_data[i].empty:
                self.training_data[i] = df
            else:
                self.training_data[i] = pd.concat([self.training_data[i], df], ignore_index=True)
            self.training_data[i].drop_duplicates(subset = 'DateTime', keep = 'first', inplace = True)
            self.training_data[i] = self.training_data[i].set_index('DateTime').asfreq('min')
            self.training_data[i].index = pd.to_datetime(self.training_data[i].index)
            # freq = pd.infer_freq(self.training_data[i].index)
            # self.training_data[i].index = self.training_data[i].index.asfreq(freq)
        
        # fig, ax = plot_series(training_data[0])
        # ax.set_xlim([training_data[0].index.min(), training_data[0].index.max()])
        # ax.set_ylim([-1, 1])
        # fig.show()


    def train_model(self):

        # 予測器をインスタンス化する
        self.forecaster = AutoARIMA(start_p = 8, start_q = 8)

        # 学習
        for i in range(self.n_symbols):
            if self.forecaster.is_fitted:
                self.forecaster.update(self.training_data[i])
            else:
                self.forecaster.fit(self.training_data[i])
            


            
        

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


        
