from bs4 import BeautifulSoup
import urllib.request

class Crawler:

    def __init__(self, symbol):
        
        self.symbol = symbol


    def fetch_stock_data(self):

        url = f'https://kabutan.jp/stock/kabuka?code={self.symbol}&ashi=day&page=1'
        req = urllib.request.Request(url, method='GET')
        
        try:
            with urllib.request.urlopen(req) as res:
                content = res.read().decode('utf-8')
        except urllib.error.HTTPError as e:
            exit('\033[31m'+ str(e) + '\033[0m')
        except Exception as e:
            exit('\033[31m' + str(e) + '\033[0m')
            
        soup = BeautifulSoup(content, 'html.parser')
        values = soup.find_all("table", class_='stock_kabuka_dwm')

        return values
        

    def extract_first_row_data(self, values):
        
        if values:
            
            table = values[0]
            first_row = table.find('tbody').find('tr')
            
            if first_row:
                
                date_cell = first_row.find('th', scope='row')
                date_time = date_cell.find('time')
                cells = first_row.find_all('td')
                
                if len(cells) >= 4:
                    high = cells[1].text.strip()
                    low = cells[2].text.strip()
                    return high, low
                else:
                    return None, None
                
            else:
                return None, None
            
        else:
            return None, None
        
    
if __name__ == '__main__':

    crawler = Crawler('1925')
    
    values = crawler.fetch_stock_data()
    target_date, high, low = crawler.extract_first_row_data(values)

    breakpoint()

