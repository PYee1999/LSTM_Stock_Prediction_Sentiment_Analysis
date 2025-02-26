_BASE_DIR = "/Users/Preston/LSTM_Stock_Prediction_Sentiment_Analysis/code"


class LOCATIONS:
    STOCK_DATA_LOCATION = f"{_BASE_DIR}/DataPreprocessing/Data/StockData"
    SAVE_MODEL_PATH = f"{_BASE_DIR}/Models"
    SAVE_NEWS_LOCATION = f"{_BASE_DIR}/DataPreprocessing/Data/NewsData"
    SAVE_GRAPH_LOCATION = f"{_BASE_DIR}/DataPreprocessing/Data/Graphs"
    SAVE_HIST_TRAIN_LOCATION = f"{_BASE_DIR}/DataPreprocessing/Data/TrainingHistory"
    TICKER_DATA_LOCATION = f"{_BASE_DIR}/TickerData/tickers.json"

class KEYS:
    STOCK_DATA_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8', 
        'Accept-Language': 'en-US,en;q=0.5',
        'DNT': '1', # Do Not Track Request Header 
        'Connection': 'close'
    }
    FINNHUB_API = "cs8ng69r01qu0vk4g5a0cs8ng69r01qu0vk4g5ag"


class FileNames:
    def __init__(self, stock, start_date, end_date):
        self.stock = stock
        self.start_date = start_date
        self.end_date = end_date
        self.suffix = f"{stock}_{start_date}_{end_date}"

    def get_hist_stock_data_name(self):
        return f"{LOCATIONS.STOCK_DATA_LOCATION}/Hist_Price_{self.suffix}.csv"
    
    def get_archive_news_name(self):
        return f"{LOCATIONS.SAVE_NEWS_LOCATION}/Archive_{self.suffix}.csv"
    
    def get_finnhub_news_name(self):
        return f"{LOCATIONS.SAVE_NEWS_LOCATION}/FinnHub_{self.suffix}.csv"
    
    def get_twit_news_name(self):
        return f"{LOCATIONS.SAVE_NEWS_LOCATION}/Twitter_{self.suffix}.csv"
    
    def get_graph_name(self):
        return f"{LOCATIONS.SAVE_GRAPH_LOCATION}/Pred_Graph_{self.suffix}.png"

    def get_hist_train_name(self):
        return f"{LOCATIONS.SAVE_HIST_TRAIN_LOCATION}/Hist_Train_{self.suffix}.csv"
    
    def get_saved_model_name(self):
        return f"{LOCATIONS.SAVE_MODEL_PATH}/LSTM_{self.suffix}.weights.h5"