import pandas as pd
import requests
import warnings

from names import FileNames, KEYS
warnings.filterwarnings(action='ignore')
import datetime
import yfinance as yf


"""
Extracting Stock Market Data for a given Stock ticker and date range from Yahoo Finance
"""

def get_stock_data(ticker: str, 
                   start_date: datetime, 
                   end_date: datetime = datetime.datetime.today(), 
                   export_data: bool = False, 
                   override_export_check: bool = True) -> pd.DataFrame:
    """
    Extract stock ticker data from yfinance within given timeframe. 
    Export if desired; also check if export file already exists.
    :param ticker: stock ticker to extract
    :param start_date: when to start extracting data from
    :param end_date: when to end extracting data
    :param export_data: option of whether to export data or not
    :param override_export_check: option to extract data regardless if already exported
    :return: dataframe of extracted stock data
    """
    # Get dates formatted
    start_date_str = datetime.datetime.strftime(start_date, '%Y-%m-%d')
    end_date_str = datetime.datetime.strftime(end_date, '%Y-%m-%d')

    # Check for exported file
    export_filepath = FileNames(stock=ticker, start_date=start_date_str, end_date=end_date_str).get_hist_stock_data_name()
    if not override_export_check:
        # Extract data 
        try:
            dl_data = pd.read_csv(export_filepath)
            return dl_data
        except FileNotFoundError as e:
            print(f"File not found: {str(e)}")
            pass

    # Set timestamps
    # SOURCE: https://www.linkedin.com/pulse/analyzing-historical-stock-data-python-yahoo-finance-ali-azary-zptxe/
    start_timestamp = int(datetime.datetime.strptime(start_date_str, '%Y-%m-%d').timestamp())
    end_timestamp = int(datetime.datetime.strptime(end_date_str, '%Y-%m-%d').timestamp())

    # Create URL to access website
    url = f'https://finance.yahoo.com/quote/{ticker}/history?period1={start_timestamp}&period2={end_timestamp}&interval=1d&frequency=1d'
    print("URL:", url)

    # Download historical data
    print(f"Downloading {ticker} data...")
    response = requests.get(url, headers=KEYS.STOCK_DATA_HEADERS)
    # print(response)

    dl_data = pd.read_html(response.text)[0]
    dl_data.set_index('Date', inplace=True)
    dl_data = dl_data[~dl_data.index.str.startswith('*')]
    print(f"dl_data:\n{dl_data}")
    dl_data.index = pd.to_datetime(dl_data.index, format='%b %d, %Y')
    dl_data = dl_data.apply(pd.to_numeric, errors='coerce')
    dl_data = dl_data.sort_values(by='Date') 
    dl_data.dropna(inplace=True)
    dl_data.reset_index(inplace=True)

    print("Data Extracted!")
    print("Dimension:", dl_data.shape)

    # Reformat closing column names
    dl_data.rename(columns={
        "Close Close price adjusted for splits.": "Close", 
        "Adj Close Adjusted close price adjusted for splits and dividend and/or capital gain distributions.": "Adj Close"
    }, inplace=True)

    # Convert all column names to lowercase
    dl_data.columns = dl_data.columns.str.lower()

    # Export data if requested
    if export_data:
        dl_data.to_csv(export_filepath)

    # Return data
    return dl_data


def get_stock_data_v2(ticker: str, 
                   start_date: datetime, 
                   end_date: datetime = datetime.datetime.today(), 
                   export_data: bool = False, 
                   override_export_check: bool = True) -> pd.DataFrame:
    """
    Extract stock ticker data from yfinance within given timeframe. 
    Export if desired; also check if export file already exists.
    Columns needed are: Date, Open, Close, High, Low, Volume
    :param ticker: stock ticker to extract
    :param start_date: when to start extracting data from
    :param end_date: when to end extracting data
    :param export_data: option of whether to export data or not
    :param override_export_check: option to extract data regardless if already exported
    :return: dataframe of extracted stock data
    """
    # Get dates formatted
    start_date_str = datetime.datetime.strftime(start_date, '%Y-%m-%d')
    end_date_str = datetime.datetime.strftime(end_date, '%Y-%m-%d')

    # Check for exported file
    export_filepath = FileNames(stock=ticker, start_date=start_date_str, end_date=end_date_str).get_hist_stock_data_name()
    if not override_export_check:
        # Extract data 
        try:
            dl_data = pd.read_csv(export_filepath)
            return dl_data
        except FileNotFoundError as e:
            print(f"File not found: {str(e)}. Need to load data.")
            pass
    
    # Download yfinance data
    dl_data = yf.download(tickers=ticker, 
                          start=start_date_str,
                          end=end_date_str, 
                          multi_level_index=False)
    
    # Clean up data
    dl_data = dl_data.sort_values(by='Date') 
    dl_data.dropna(inplace=True)
    dl_data.reset_index(inplace=True)
    dl_data.columns = dl_data.columns.str.lower()

    print(dl_data)

    # Export data if requested
    if export_data:
        dl_data.to_csv(export_filepath)

    # Return data
    return dl_data



# print(get_stock_data(ticker="AAPL", 
#                      start_date=datetime.datetime(2022, 1, 1), 
#                      export_data=True))