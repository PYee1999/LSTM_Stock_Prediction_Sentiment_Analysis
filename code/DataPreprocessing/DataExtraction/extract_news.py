"""
Extracting news sentiment data on specific company stock
"""

# FinnHub Resources:
# https://finnhub.io/docs/api/websocket-news (Premium)
# https://finnhub.io/docs/api/company-news

import finnhub
import datetime
import string
import re
import os
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

from tqdm import tqdm
from names import FileNames

from internetarchive.session import ArchiveSession
from internetarchive.search import Search
from internetarchive import get_item
from langdetect import detect


FINNHUB_API = os.getenv("FINNHUB_API")
CLEANR = re.compile('<.*?>')


class NewsExtraction:
    def __init__(self, 
                 ticker: str, 
                 start_date: datetime, 
                 end_date: datetime, 
                 export_data: bool = True, 
                 always_extract: bool = True):
        """
        :param ticker: stock ticker string
        :param start_date: earliest date to extract stock news data from
        :param end_date: latest date to extract stock news data from
        :param export_data: option to export extracted news data to csv file
        :param always_extract: option to always extract data from source instead of getting from .csv
        """
        self.ticker = ticker
        self.start_date = start_date.strftime("%Y-%m-%d")
        self.end_date = end_date.strftime("%Y-%m-%d")
        self.export_data = export_data
        self.always_extract = always_extract

        self.name = ""
        self.extracted_data = None
        self.export_filepath = None
        self.extract_columns = []

    def _extract_if_exists(self):
        """
        Option to extract sentiment data from existing file instead of reading it directly
        """
        # Extract data from csv file
        if not self.always_extract:
            try:
                self.extracted_data = pd.read_csv(self.export_filepath)
            except FileNotFoundError as e:
                print(f"File not found: {str(e)}")
                pass

    def _export(self):
        """
        Option to export extracted data to csv
        """
        if self.export_data:
            self.extracted_data.to_csv(self.export_filepath, index=False)
            print(f"{self.name} news data successfully exported")

    def get_news(self):
        """
        Extract news data for given source
        """
        pass


class FinnHubNews(NewsExtraction):
    def __init__(self, 
                 ticker: str, 
                 start_date: datetime, 
                 end_date: datetime, 
                 export_data: bool = True, 
                 always_extract: bool = True):
        super().__init__(ticker, start_date, end_date, export_data, always_extract)

        self.name = "FinnHub"
        self.export_filepath = FileNames(stock=self.ticker, 
                                         start_date=self.start_date, 
                                         end_date=self.end_date).get_finnhub_news_name()
        self.extract_columns = ["datetime", "headline", "summary"]

    def get_news(self):
        # Return news data if already downloaded beforehand
        self._extract_if_exists()
        if self.extracted_data is not None and len(self.extracted_data) > 0:
            return self.extracted_data

        # Extract news from source
        finnhub_client = finnhub.Client(api_key=FINNHUB_API)
        results = finnhub_client.company_news(self.ticker, _from=self.start_date, to=self.end_date)

        # Convert results into dataframe
        results_df = pd.DataFrame(results)
        
        # Extract dataframe columns
        results_df = results_df[self.extract_columns]
        
        # Convert date from UNIX to datetime and rename columns
        print(f"FinnHub News Dataframe:\n{results_df}")
        results_df['datetime'] = pd.to_datetime(results_df['datetime'], unit='s', errors='coerce')
        results_df = results_df.rename(columns={
            "datetime": "date", 
            "headline": "title"
        })

        # Filter out empty rows
        results_df.dropna(inplace=True)

        # Remove timestamps
        results_df['date'] = results_df['date'].dt.date

        self.extracted_data = results_df

        # Export results if requested
        self._export()

        st.write(f"{self.name} news data successfully downloaded!")

        # Return dataframe
        return self.extracted_data


class InternetArchiveNews(NewsExtraction):
    def __init__(self, 
                 ticker: str, 
                 start_date: datetime, 
                 end_date: datetime, 
                 export_data: bool = True, 
                 always_extract: bool = True,
                 limit = None):
        super().__init__(ticker, start_date, end_date, export_data, always_extract)

        self.name = "InternetArchive"
        self.export_filepath = FileNames(stock=self.ticker, 
                                         start_date=self.start_date, 
                                         end_date=self.end_date).get_archive_news_name()
        self.extract_columns = ["publicdate", "title", "description"]
        self.limit = limit
    
    def get_news(self):
        # Return news data if already downloaded beforehand
        self._extract_if_exists()
        if self.extracted_data is not None and len(self.extracted_data) > 0:
            return self.extracted_data

        # Get short and long name of stock ticker
        pg = yf.Ticker(ticker=self.ticker)
        long_name = pg.info['longName']
        long_name = long_name.translate(str.maketrans('', '', string.punctuation))
        long_name = long_name.split(' ', 1)[0]

        # Create search and extract results
        s = ArchiveSession()
        search = Search(s, f'title:({self.ticker} OR {long_name}) AND date:[{self.start_date} TO {self.end_date}]')
        print("Search Query:", search)
        num_results = len(search)
        if num_results == 0:
            raise Exception(f"No results found for {self.ticker} between date range of {self.start_date} and {self.end_date}")
        else:
            print(f"Total results found: {num_results}")

        # Set up dataframe and counts
        extracted_data = pd.DataFrame(columns=self.extract_columns)
        missing = 0
        count = 0

        # Loop through every entry extracted in search
        progress_bar = st.progress(0, f"Extracting sentiment for {self.ticker}")
        i = 0
        total_searches = len(search)
        prg = 0
        for result in tqdm(search):
            # Calculate progress bar
            i += 1
            prg = i/total_searches
            progress_bar.progress(prg, f"Extracting {i}/{total_searches} ({format(prg * 100, ".2f")}%) search results (This may take a while; Do not refresh page)")

            if self.limit:
                # If a limit exists, then keep count and break when count meets limit
                if count >= self.limit:
                    break
                count += 1
            try:
                # Extract needed columns and append/concat to extracted_data
                item_data = get_item(result["identifier"]).metadata
                for col in self.extract_columns:
                    if col not in item_data.keys():
                        item_data[col] = np.nan

                dict_you_want = {key: [item_data[key]] for key in self.extract_columns}
                dict_you_want = pd.DataFrame(dict_you_want)
                extracted_data = pd.concat([extracted_data, dict_you_want])
            except Exception as e:
                # Count missing entries due to missing columns
                missing += 1
                pass
        progress_bar.progress(prg, "Search results extraction complete!")

        # Show missing entries count
        missing_pct = missing/num_results
        print(f"Missed entries: {missing} ({missing_pct:.2%})")

        # Filter for English-only Descriptions, Clean out HTML tags, and rename to summary
        extracted_data = extracted_data[extracted_data['description'].apply(self._is_english)]
        extracted_data['description'] = extracted_data['description'].apply(self._cleanhtml)

        # Rename, sort and format date column
        extracted_data = extracted_data.rename(columns={'publicdate': 'date', 'description': 'summary'})
        extracted_data['date'] = pd.to_datetime(extracted_data['date'])
        extracted_data = extracted_data.sort_values(by='date')
        extracted_data['date'] = extracted_data['date'].dt.strftime('%Y-%m-%d')

        # Clean out invalid descriptions
        extracted_data = self._clean_archive_desc(extracted_data)
        self.extracted_data = extracted_data

        # Export results if requested
        self._export()

        st.write(f"{self.name} news data successfully downloaded!")

        # Return dataframe
        return self.extracted_data

    def _clean_archive_desc(self, df: pd.DataFrame, desc_col: str = "summary"):
        """
        Clean out invalid description in archive data
        :param data: pandas dataframe of data containing
        :param desc_col: column of summary/description
        """
        df.loc[df[desc_col].str.contains('Perma.cc archive'), desc_col] = np.nan
        return df

    def _cleanhtml(self, raw_html_str):
        """
        Cleans raw HTML string for the normal text
        """
        cleantext = re.sub(CLEANR, '', raw_html_str)
        return cleantext

    def _is_english(self, text):
        """
        Checks/Validates if the text is in English
        """
        try:
            return detect(text) == 'en'
        except:
            return False  # Treat detection failures as non-English