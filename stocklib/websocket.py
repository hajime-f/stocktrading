import websockets
import asyncio
import json
import traceback

class WebSocketStreamProcessor:

    def __init__(self, ip_address, port):

        self.uri = "ws://" + ip_address + ":" + port + "/kabusapi/websocket"

    def __call__(self, func):

        async def stream(func):
            
            async with websockets.connect(self.uri, ping_timeout=None) as ws:
                while not ws.close_reason is not None:
                    response = await ws.recv()
                    try:
                        func(json.loads(response))
                    except BaseException as e:
                        print(e)
                        traceback.print_exc()
                        self.loop.stop()
            
        self.loop = asyncio.get_event_loop()
        self.loop.create_task(stream(func))
        return stream

    def run(self):
        self.loop.run_forever()


        
