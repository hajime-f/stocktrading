import pandas as pd

from data_management import DataManagement

pd.set_option("display.max_rows", None)


class LSTM:
    def __init__(self):
        pass


if __name__ == "__main__":
    dm = DataManagement()
    df = dm.load_stock_data(1301)

    # 翌営業日の終値が当日より2.5%以上上昇していたらフラグを立てる
    df_shift = df.shift(-1)
    df["increase"] = 0
    df.loc[df_shift["close"] > df["close"] * 1.025, "increase"] = 1

    # 移動平均線を追加する
    df_ma5 = df["close"].rolling(window=5).mean()
    df_ma25 = df["close"].rolling(window=25).mean()
    df["diff_ma"] = df_ma5 - df_ma25

    # MACDを追加する
    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["SIGNAL"] = df["MACD"].ewm(span=9).mean()
    df["HISTOGRAM"] = df["MACD"] - df["SIGNAL"]

    # ボリンジャーバンドを追加する
    sma20 = df["close"].rolling(window=20).mean()
    std20 = df["close"].rolling(window=20).std()
    df["Upper"] = sma20 + (std20 * 2)
    df["Lower"] = sma20 - (std20 * 2)

    # RSIを追加する
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # 終値の前日比を追加する
    df_shift = df.shift(1)
    df["close_rate"] = (df["close"] - df_shift["close"]) / df_shift["close"]

    # 始値と終値の差を追加する
    df["trunk"] = df["open"] - df["close"]

    lstm = LSTM()
    breakpoint()
