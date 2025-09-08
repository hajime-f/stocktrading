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
    list_data = []

    for _, row in dm.fetch_target(table_name="Target").iterrows():
        
        crawler = Crawler(row["code"])
        data = crawler.extract_todays_data(crawler.fetch_stock_data())

        try:
            open_price = float(data[0].replace(",", ""))
            close_price = float(data[3].replace(",", ""))

            list_data.append(
                [
                    row["date"],
                    row["code"],
                    row["brand"],
                    open_price,
                    close_price,
                    row["pred"],
                    row["side"],
                ]
            )
        except Exception:
            pass

    df = pd.DataFrame(
        list_data, columns=["date", "code", "brand", "open", "close", "pred", "side"]
    )

    total_price = 0
    for index, row in df.iterrows():
        if row["side"] == 1:
            change = row["open"] - row["close"]
            total_price += row["close"]
        elif row["side"] == 2:
            change = row["close"] - row["open"]
            total_price += row["open"]
        else:
            raise ValueError("売買フラグが不正です。")
        df.loc[index, "change"] = change

    df = df[["date", "code", "brand", "open", "close", "change", "pred", "side"]]

    total_pl = df["change"].sum()
    print(f"{int(total_price * 100):,} 円かけて {int(total_pl * 100):,} 円")
    # dm.save_profit_loss(df, table_name="ProfitLoss3")
