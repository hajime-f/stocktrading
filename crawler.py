from bs4 import BeautifulSoup
import urllib.request

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

    total_change = 0
    total_open_price = 0

    # 予測モデルが提案するすべての銘柄を買った場合
    for symbol in symbols:
        crawler = Crawler(symbol[0])
        values = crawler.fetch_stock_data()
        data = crawler.extract_todays_data(values)

        open_price = float(data[0].replace(",", ""))
        close_price = float(data[3].replace(",", ""))
        change = float(data[4])
        change_p = float(data[5])

        print(f"{symbol[0]}：{symbol[1]}, {change}, {change_p}％：{symbol[2]:.3f}")
        try:
            total_open_price += open_price
            total_change += close_price - open_price
        except ValueError:
            continue

    print(
        f"All：{int(total_open_price) * 100:,} で {int(total_change) * 100:,} の損益\n"
    )

    total_change = 0
    total_open_price = 0

    # 予測値が 0.8 以上の銘柄を買った場合
    for symbol in symbols:
        if symbol[2] < 0.7:
            continue

        crawler = Crawler(symbol[0])
        values = crawler.fetch_stock_data()
        data = crawler.extract_todays_data(values)

        open_price = float(data[0].replace(",", ""))
        close_price = float(data[3].replace(",", ""))
        change = float(data[4])
        change_p = float(data[5])

        print(f"{symbol[0]}：{symbol[1]}, {change}, {change_p}％：{symbol[2]:.3f}")
        try:
            total_open_price += open_price
            total_change += close_price - open_price
        except ValueError:
            continue

    print(
        f"0.7：{int(total_open_price) * 100:,} で {int(total_change) * 100:,} の損益\n"
    )

    total_change = 0
    total_open_price = 0

    # 予測値が 0.8 以上の銘柄を買った場合
    for symbol in symbols:
        if symbol[2] < 0.8:
            continue

        crawler = Crawler(symbol[0])
        values = crawler.fetch_stock_data()
        data = crawler.extract_todays_data(values)

        open_price = float(data[0].replace(",", ""))
        close_price = float(data[3].replace(",", ""))
        change = float(data[4])
        change_p = float(data[5])

        print(f"{symbol[0]}：{symbol[1]}, {change}, {change_p}％：{symbol[2]:.3f}")
        try:
            total_open_price += open_price
            total_change += close_price - open_price
        except ValueError:
            continue

    print(
        f"0.8：{int(total_open_price) * 100:,} で {int(total_change) * 100:,} の損益\n"
    )

    total_change = 0
    total_open_price = 0

    # 予測値が 0.9 以上の銘柄を買った場合
    for symbol in symbols:
        if symbol[2] < 0.9:
            continue

        crawler = Crawler(symbol[0])
        values = crawler.fetch_stock_data()
        data = crawler.extract_todays_data(values)

        open_price = float(data[0].replace(",", ""))
        close_price = float(data[3].replace(",", ""))
        change = float(data[4])
        change_p = float(data[5])

        print(f"{symbol[0]}：{symbol[1]}, {change}, {change_p}％：{symbol[2]:.3f}")
        try:
            total_open_price += open_price
            total_change += close_price - open_price
        except ValueError:
            continue

    print(f"0.9：{int(total_open_price) * 100:,} で {int(total_change) * 100:,} の損益")
