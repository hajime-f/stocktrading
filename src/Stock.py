



class Stock():

    def __init__(self, code, key, market=1, db_file_name='stocks.db', log_path_name='./logs/'):

        self.code = code
        self.key = key
        self.market = market
        self.db_file_name = db_file_name
        self.log_path_name = log_path_name

        content = key.fetch_code_info(code, 1)
        self.disp_name = content["DisplayName"]
        self.unit = int(content["TradingUnit"])
        




