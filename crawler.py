from bs4 import BeautifulSoup
import urllib.request

from data_management import DataManagement


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

    def extract_change_data(self, values):
        if values:
            table = values[0]
            first_row = table.find("tbody").find("tr")

            if first_row:
                cells = first_row.find_all("td")

                if len(cells) >= 6:
                    change = cells[4].text.strip()
                    change_p = cells[5].text.strip()
                    return change, change_p
                else:
                    return None, None

            else:
                return None, None

        else:
            return None, None


if __name__ == "__main__":
    dm = DataManagement()
    symbols = [[symbol[1], symbol[2]] for symbol in dm.fetch_target()]

    total = 0

    for symbol in symbols:
        crawler = Crawler(symbol[0])
        values = crawler.fetch_stock_data()
        change, change_p = crawler.extract_change_data(values)

        print(f"{symbol[0]}：{symbol[1]}, {change}, {change_p}％")
        try:
            total += float(change)
        except ValueError:
            continue

    print(f"合計：{total}")
