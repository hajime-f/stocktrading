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
    table_name = ["Target_Long", "Target_Short"]
    pl_name = ["PL_Long", "PL_Short"]

    for table, pl in zip(table_name, pl_name):
        symbols = [[p[1], p[2], p[3]] for p in dm.fetch_target(table_name=table)]
        list_data = []

        for s in symbols:
            crawler = Crawler(s[0])
            values = crawler.fetch_stock_data()
            data = crawler.extract_todays_data(values)

            try:
                open_price = float(data[0].replace(",", ""))
                close_price = float(data[3].replace(",", ""))

                list_data.append(
                    [
                        datetime.now().strftime("%Y-%m-%d"),
                        s[0],
                        s[1],
                        open_price,
                        close_price,
                        s[2],
                    ]
                )
            except Exception:
                pass

        df = pd.DataFrame(
            list_data, columns=["date", "code", "brand", "open", "close", "pred"]
        )
        if table_name == "Target_Long":
            df["change"] = df["close"] - df["open"]
            total_price = df["open"].sum()
        else:
            df["change"] = df["open"] - df["close"]
            total_price = df["close"].sum()
        df = df[["date", "code", "brand", "open", "close", "change", "pred"]]

        total_pl = df["change"].sum()
        if table_name == "Target_Long":
            print("買い建て")
        else:
            print("売り建て")
        print(f"{int(total_price * 100):,} 円かけて {int(total_pl * 100):,} 円")

        dm.save_profit_loss(df, table_name=pl)
