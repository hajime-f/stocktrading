import stocklib

if __name__ == '__main__':
    
    lib = stocklib.Initialize()

    @lib.websocket
    def receive(msg):
        print(msg)

    lib.websocket.run()
    
    
    
    
    
