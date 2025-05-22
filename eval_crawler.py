import urllib.request
from datetime import datetime, date

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

    today = date.today().strftime("%Y-%m-%d")
    # df = dm.load_table_by_date("Evaluation", today)
    df = dm.load_table_by_date("Evaluation3", today)

    trial = df["trial"].max().item()
    total = []

    for i in range(1, trial + 1):
        df_trial = df.loc[df["trial"] == i, :]
        change = []

        for index, row in df_trial.iterrows():
            crawler = Crawler(row["code"])
            values = crawler.fetch_stock_data()
            data = crawler.extract_todays_data(values)

            try:
                open_price = float(data[0].replace(",", ""))
                close_price = float(data[3].replace(",", ""))
            except Exception:
                pass

            if row["side"] == 1:
                change.append(open_price - close_price)
            else:
                change.append(close_price - open_price)

        total.append(int(sum(change) * 100))

    print(f"{sum(total) / len(total)} 円")
