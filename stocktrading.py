import stocklib

if __name__ == '__main__':
    
    lib = stocklib.Initialize()

    # 預金残高（現物の買付余力）を問い合わせる
    deposit = lib.information.deposit()
    print(f"\033[33m預金残高：{int(deposit):,} 円\033[0m")
    
    @lib.websocket
    def receive(msg):
        print(msg)

    lib.websocket.run()
    
    
    
    
    
