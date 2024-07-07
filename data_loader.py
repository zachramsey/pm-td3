import os
import sys
import platformdirs as ad

import pandas as pd
import pandas_ta as ta
import pandas_market_calendars as mcal
from sklearn.preprocessing import StandardScaler

import yfinance as yf
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter

# ==================================================================================================
class DataLoader:
    '''### Fetch historical stock data and extract features for a given list of stock symbols
    Args:
        symbols (list): A mixed list of file paths and/or stock symbols
        data_dir (str): The directory to store the historical stock data
    Attributes:
        nyse (MarketCalendar): The NYSE calendar for valid trading days
        scaler (StandardScaler): The Standard Scaler for normalization
        symbols (list): A list of stock symbols
        data (dict): A dictionary to store the historical stock data
        features (dict): A dictionary to store the extracted features
    Methods:
        get_data(symbol) -> pd.DataFrame: Get the historical stock data for a given symbol
        get_features(symbol) -> pd.DataFrame: Get the extracted features for a given symbol
        get_symbols() -> list: Get the list of stock symbols
        refresh_data() -> None: Refresh the historical stock data for each symbol in the symbols list
    '''

    # --------------------------------------------------------------------------------------------
    def __init__(self, symbols, data_dir='stock_data'):
        _parent_dp = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        _src_dp = _parent_dp
        sys.path.insert(0, _src_dp)

        # Use adjacent cache folder for testing, delete if already exists and older than today
        testing_cache_dirpath = os.path.join(ad.user_cache_dir(), "py-yfinance-testing")
        yf.set_tz_cache_location(testing_cache_dirpath)
        if os.path.isdir(testing_cache_dirpath):
            mtime = pd.Timestamp(os.path.getmtime(testing_cache_dirpath), unit='s', tz='UTC')
            if mtime.date() < pd.Timestamp.now().date():
                import shutil
                shutil.rmtree(testing_cache_dirpath)

        # Setup a session to rate-limit and cache persistently:
        class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
            pass
        history_rate = RequestRate(1, Duration.SECOND*1.8)
        limiter = Limiter(history_rate)
        cache_fp = os.path.join(testing_cache_dirpath, "unittests-cache")
        self.session = CachedLimiterSession(
            limiter=limiter,
            bucket_class=MemoryQueueBucket,
            backend=SQLiteCache(cache_fp, expire_after=pd.Timedelta(hours=1)),
        )

        self.nyse = mcal.get_calendar('NYSE')   # NYSE calendar for valid trading days
        self.scaler = StandardScaler()          # Standard Scaler for normalization
        self.symbols = symbols                  # List of stock symbols

        self.data = {}                          # Dictionary to store the historical stock data
        self.features = {}                      # Dictionary to store the extracted features

        self.data_dir = data_dir                    # Directory to store the historical stock data
        self.symbols = self._load_symbols(symbols)  # Load the stock symbols
        self._load_data()                           # Load the historical stock data
        self._calculate_features()                  # Calculate the features for each symbol

    # --------------------------------------------------------------------------------------------
    def _load_symbols(self, symbols) -> list:
        '''### Generate a list of stock symbols
        Args:
            symbols (list): A mixed list of file paths and/or stock symbols
        Returns:
            list: A list of stock symbols
        '''
        all_symbols = []
        for symbol in symbols:
            if os.path.exists(symbol):
                all_symbols.extend(pd.read_csv(symbol, header=None)[0].tolist())
            else:
                all_symbols.append(symbol)
        return all_symbols

    # --------------------------------------------------------------------------------------------
    def _load_data(self) -> None:
        '''### Fetch historical stock data for each symbol in the symbols list
        Args:
            data_dir (str): The directory to store the historical stock data
            symbols (list): A list of stock symbols
        '''
        print(f'{"-"*50}\nFetching Historical Data...')
        
        # Fetch the historical data for each symbol
        for i, symbol in enumerate(self.symbols):

            # Create the data directory if it does not exist
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)

            # Fetch or update the historical data for the given symbol
            file_path = os.path.join(self.data_dir, f"{symbol}.csv")
            if os.path.exists(file_path):
                self._update_data(file_path, symbol)    # Update the existing data
            else:
                self._fetch_data(file_path, symbol)     # Fetch the entire data 

            print(f'{i+1} / {len(self.symbols)} fetched | {(i+1)/len(self.symbols)*100:.2f}% complete', end='\r')
        print(f'\nHistorical data successfully updated.\n{"-"*50}')
    
    # --------------------------------------------------------------------------------------------
    def _fetch_data(self, file_path, symbol) -> None:
        '''### Fetch the historical stock data for a given symbol
        Args:
            file_path (str): The file path to store the historical stock data
            symbol (str): The stock symbol
        '''
        # Fetch the entire historical data for a given symbol
        df = yf.download(symbol, repair=True, progress=False, session=self.session)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        # convert the index to datetimeindex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        df.to_csv(file_path)                # Save the data to a CSV file
        self.data[symbol] = df              # Store the data in the data dictionary
    
    # --------------------------------------------------------------------------------------------
    def _update_data(self, file_path, symbol) -> None:
        '''### Update the existing historical stock data for a given symbol
        Args:
            file_path (str): The file path to the existing historical stock data
            symbol (str): The stock symbol
        '''
        df = pd.read_csv(file_path, index_col='Date')   # Load the existing data
        
        # convert the index to datetimeindex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        last_record_date = df.index[-1].date()
        last_trading_date = self.nyse.valid_days(start_date=df.index[-1], end_date=pd.Timestamp.now())[-1].date()

        # Check if the data is up-to-date
        if last_record_date < last_trading_date:
            new_data = yf.download(symbol, start=df.index[-1]+pd.Timedelta(days=1), progress=False, session=self.session)
            new_data = new_data[['Open', 'High', 'Low', 'Close', 'Volume']]
            print(new_data)
            new_data.to_csv(file_path, mode='a', header=False)    # Append the new data to the existing data
            df = pd.concat([df, new_data])                  # Concatenate the existing and new data

        self.data[symbol] = df    # Store the updated data in the data dictionary
    
    # --------------------------------------------------------------------------------------------
    def _calculate_features(self) -> None:
        '''
        ### Extract features from the historical stock data for a symbol
        Args:
            symbol (list): The stock symbols to extract the features for
        '''
        print("Extracting Features...")
        for i, symbol in enumerate(self.symbols):
            df = self.data[symbol].copy()   # Load the historical stock data

            # Moving Averages
            df['SMA_20'] = ta.sma(df['Close'], length=20)
            df['SMA_50'] = ta.sma(df['Close'], length=50)
            df['EMA_20'] = ta.ema(df['Close'], length=20)
            df['EMA_50'] = ta.ema(df['Close'], length=50)

            # Volitility Indicators
            bbands = ta.bbands(df['Close'], length=20, std=2)
            df['UpperBB'] = bbands['BBU_20_2.0']
            df['MiddleBB'] = bbands['BBM_20_2.0']
            df['LowerBB'] = bbands['BBL_20_2.0']
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

            # Momentum Indicators
            df['RSI'] = ta.rsi(df['Close'], length=14)
            macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
            df['MACD'] = macd['MACD_12_26_9']

            # Volume-based Indicators
            df['OBV'] = ta.obv(df['Close'], df['Volume'])
            df['CMF'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=21)
            df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'], length=14)

            # Drop the NaN values and normalize the data
            df.dropna(inplace=True)
            df = df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
            # df = self.scaler.fit_transform(df)
            df = pd.DataFrame(df, columns=df.columns, index=df.index)

            self.features[symbol] = df    # Store the extracted features in the features dictionary

            print(f'{i+1} / {len(self.symbols)} extracted | {(i+1)/len(self.symbols)*100:.2f}% complete', end='\r')
        print(f'\nFeatures successfully extracted.\n{"-"*50}')

    # --------------------------------------------------------------------------------------------
    def get_valid_days(self, start_date, end_date) -> pd.DatetimeIndex:
        '''### Get the valid trading days for a given symbol
        Args:
            symbol (str): The stock symbol
            start_date (str): The start date
            end_date (str): The end date
        Returns:
            DatetimeIndex: The valid trading days for the given symbol
        '''
        return self.nyse.valid_days(start_date=start_date, end_date=end_date)
    
    # --------------------------------------------------------------------------------------------
    def get_data(self, symbols, start_date=None, end_date=None) -> pd.DataFrame:
        '''### Get the historical stock data for given symbols from start to end date
        Args:
            symbols (str, list): The stock symbol or list of stock symbols
            start_date (str): The start date for the historical data
            end_date (str): The end date for the historical data
        Returns:
            dict: The historical stock data for the given symbols
        '''
        if isinstance(symbols, str):
            symbols = [symbols]

        if start_date is None:
            start_date = max([self.data[symbol].index[0].date() for symbol in symbols])

        if end_date is None:
            end_date = pd.Timestamp.now().date()

        return {symbol: self.data[symbol].loc[start_date:end_date] for symbol in symbols}
    
    # --------------------------------------------------------------------------------------------
    def get_features(self, symbols, start_date=None, end_date=None) -> pd.DataFrame:
        '''### Get the extracted features for given symbols from start to end date
        Args:
            symbols (str, list): The stock symbol or list of stock symbols
            start_date (str): The start date for the extracted features
            end_date (str): The end date for the extracted features
        Returns:
            dict: The extracted features for the given symbols
        '''
        if isinstance(symbols, str):
            symbols = [symbols]

        if start_date is None:
            start_date = max([self.features[symbol].index[0].date() for symbol in symbols])

        if end_date is None:
            end_date = pd.Timestamp.now().date()

        return {symbol: self.features[symbol].loc[start_date:end_date] for symbol in symbols}
        
    # --------------------------------------------------------------------------------------------
    def get_price(self, symbol, date) -> float:
        '''### Get the stock price for a given symbol and date
        Args:
            symbol (str): The stock symbol
            date (str): The date
        Returns:
            float: The stock price for the given symbol and date
        '''
        return self.data[symbol].loc[date, 'Close']

    # --------------------------------------------------------------------------------------------
    def get_symbols(self) -> list:
        '''### Get the list of stock symbols
        Returns:
            list: The list of stock symbols
        '''
        return self.symbols
    
    # --------------------------------------------------------------------------------------------
    def refresh_data(self) -> None:
        '''### Refresh the historical stock data for each symbol in the symbols list'''
        self.data = self._load_data(self.data_dir, self.symbols)
        self._calculate_features()
    
# =================================================================================================