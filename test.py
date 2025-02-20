import time
from library import StockLibrary

if __name__ == '__main__':

    # 株ライブラリを初期化する
    lib = StockLibrary()

    # 登録銘柄リストからすべての銘柄を削除する
    lib.unregister_all()

    # 銘柄登録
    lib.register([1475,])

    # 成行で買い注文を出す
    content = lib.buy_at_market_price_with_cash(1475, 10)

    order_result = content['Result']
    if order_result == 0:
        buy_order_id = content['OrderId']
        console.log(f"[blue]成行で買い注文を出しました[/] \U0001F4B8")
    
    # 買い注文の約定状況を確認する
    result = lib.check_execution(buy_order_id)

    while result['State'] != 5:
        console.log(f"[blue]買い注文がまだ約定していません[/] \U0001F4B8")
        result = lib.check_execution(buy_order_id)
        time.sleep(1)
        
    console.log(f"[blue]買い注文が約定しました[/] \U0001F4B0")
    
