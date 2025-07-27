class StockTradingError(Exception):
    pass


class ConfigurationError(StockTradingError):
    pass


class APIError(StockTradingError):
    pass


class DataProcessingError(StockTradingError):
    pass


class UnexpectedOrderCountError(StockTradingError):
    pass
