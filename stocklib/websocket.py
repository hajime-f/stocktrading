import websockets
import asyncio
import json
import traceback

class WebSocketStreamProcessor:

    def __init__(self, ip_address, port, timeout_sec=3600, ping_interval=60):

        self.uri = "ws://" + ip_address + ":" + port + "/kabusapi/websocket"
        self.timeout_sec = timeout_sec
        self.ping_interval = ping_interval
        self.closed = asyncio.Event()

        
    def __call__(self, func):

        async def stream(func):
            
            try:
                async with websockets.connect(self.uri, ping_timeout=self.timeout_sec) as ws:

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


    def set_model(self, model):
        self.model = model

    
    def run(self):

        async def wait_and_train():
            await self.closed.wait()  # 接続が閉じるまで待つ
            print("学習を開始します。")
            self.model.train.train_model()
            print("学習が完了しました。")
            exit()

        self.loop.create_task(wait_and_train())
        self.loop.run_forever()
