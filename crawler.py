import urllib.request
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup

from data_manager import DataManager


class Crawler:
    def __init__(self, symbol):
        self.symbol = symbol

    def fetch_stock_data(self):
        url = f"https://kabutan.jp/stock/kabuka?code={self.symbol}&ashi=day&page=1"
        req = urllib.request.Request(url, method="GET")

        try:
            with urllib.request.urlopen(req) as res:
                content = res.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            exit("\033[31m" + str(e) + "\033[0m")
        except Exception as e:
            exit("\033[31m" + str(e) + "\033[0m")

        soup = BeautifulSoup(content, "html.parser")
        values = soup.find_all("table", class_="stock_kabuka0")

        return values

    def extract_todays_data(self, values):
        if values:
            table = values[0]
            first_row = table.find("tbody").find("tr")

            if first_row:
                cells = first_row.find_all("td")

                if len(cells) >= 6:
                    data = [c.text.strip() for c in cells]
                    return data
                else:
                    return None

            else:
                return None

        else:
            return None


if __name__ == "__main__":
    dm = DataManager()
    symbols = [[symbol[1], symbol[2], symbol[3]] for symbol in dm.fetch_target()]

    list_data = []

    for symbol in symbols:
        crawler = Crawler(symbol[0])
        values = crawler.fetch_stock_data()
        data = crawler.extract_todays_data(values)

        open_price = float(data[0].replace(",", ""))
        close_price = float(data[3].replace(",", ""))

        list_data.append(
            [
                datetime.now().strftime("%Y-%m-%d"),
                symbol[0],
                symbol[1],
                open_price,
                close_price,
                symbol[2],
            ]
        )

    df = pd.DataFrame(
        list_data, columns=["date", "code", "brand", "open", "close", "pred"]
    )
    df["change"] = df["close"] - df["open"]

    # 予測モデルが提案するすべての銘柄を買った場合
    total = df["change"].sum()
    print(f"全部：{int(total * 100):,} 円")

    # 予測値が 0.80 以上の銘柄を買った場合
    total = df[df["pred"] >= 0.80]["change"].sum()
    print(f"0.80：{int(total * 100):,} 円")

    # 予測値が 0.85 以上の銘柄を買った場合
    total = df[df["pred"] >= 0.85]["change"].sum()
    print(f"0.85：{int(total * 100):,} 円")

    dm.save_profit_loss(df)
