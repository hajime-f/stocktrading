from Key import Key
from Trade import init_portfolio, trade
from datetime import date, datetime, timezone, timedelta
import time, jpholiday


if __name__ == '__main__':

    # サーバにアクセスするキーをインスタンス化する
    key = Key()
    
    # ポートフォリオを初期化する
    pf = init_portfolio(key)
    
    # 土日祝判定
    JST = timezone(timedelta(hours=+9), 'JST')
    d = datetime.now(JST)
    if d.weekday() >= 5 or jpholiday.is_holiday(d):
        exit('本日は土日祝のため市場は開いていません。')
    
    # 開場と閉場の日時を得る
    start1 = datetime(year=d.year, month=d.month, day=d.day, hour=9, minute=0, tzinfo=JST)   # 9:00から
    end1 = datetime(year=d.year, month=d.month, day=d.day, hour=15, minute=0, tzinfo=JST)    # 15:00まで
    start2 = datetime(year=d.year, month=d.month, day=d.day, hour=11, minute=30, tzinfo=JST) # 昼休み11:30から
    end2 = datetime(year=d.year, month=d.month, day=d.day, hour=12, minute=30, tzinfo=JST)   # 昼休み12:30まで

    # 市場が開くまで待機する
    while datetime.now(JST) < start1:
        time.sleep(1)
    
    # 市場が開いている間、次の処理を繰り返す
    while start1 <= datetime.now(JST) <= end1:

        # 昼休みは待機する
        while start2 < datetime.now(JST) < end2:
            time.sleep(1)
            
        # 取引する
        tr = trade(pf)

