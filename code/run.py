"""
Main code

1. Given timeframe, extract news/social media data and stock data
2. Save data in locations
3. Preprocess data
4. Train data (start with random forest, then xgb boost, and then ensemble of both)
5. Save models
6. Evaluate models (run/compare X_test, y_test)
7. Make predictions (1, 2, 3, 5, 7, 10, 20 days)
    - Given today's news (using function to evaluate today's news sentiment)
    - What is the prediction for today or future?
"""

import io
import os
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd

from keras.models import Sequential # type: ignore
from keras.layers import Dense, Dropout, LSTM # type: ignore
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard # type: ignore
from keras.optimizers import Adam # type: ignore

from DataPreprocessing.DataExtraction.extract_stock_data import get_stock_data, get_stock_data_v2
from DataPreprocessing.SentimentAnalysis.sentiment import VaderSentiment, FinBertSentiment, TextBlobSentiment, FlairSentiment, average_sentiment_per_date
from DataPreprocessing.DataExtraction.extract_news import FinnHubNews, InternetArchiveNews
from names import FileNames, LOCATIONS

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
import mpld3

from pylab import rcParams


# Configure page width
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    div.block-container {
        width: 75%;
        margin: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)


class PredictionConfig():
    def __init__(self, stock, start_date, end_date, 
                 open, close, high, low, vol, 
                 vader, finbert, textblob, flair,
                 n_units, n_layers, n_epochs, batch_size, learning_rate, dropout_rate,
                 n_past=90, n_future=30):
        """
        Parameters for running predictions
        """
        # Stock ticker
        self.stock = stock

        # Start date and end date for range of stocks/sentiments to extract
        self.start_date = start_date
        self.end_date = end_date

        # Toggles for extracting stock data
        self.open_tog_on = open
        self.close_tog_on = close
        self.high_tog_on = high
        self.low_tog_on = low
        self.vol_tog_on = vol
        self.stock_data_options = {
            "open": self.open_tog_on,
            "close": self.close_tog_on,
            "high": self.high_tog_on,
            "low": self.low_tog_on,
            "volume": self.vol_tog_on,
        }

        # Toggles for extracting sentiment data
        self.vader_on = vader
        self.finbert_on = finbert
        self.textblob_on = textblob
        self.flair_on = flair

        # LSTM Model Hyperparameters
        self.n_units = n_units
        self.n_layers = n_layers - 2  # Remove two to consider any extra layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout_rate

        # Number of days to look back (n_past) and predict in the future (n_future)
        self.n_past = n_past
        self.n_future = n_future


class StockPricePredictionSentiment():
    def __init__(self, config):
        self.pred_config = config
        self.training_size = 0.8
        self.start_date_for_plotting = self.pred_config.start_date

        self.file_location = FileNames(
            stock=self.pred_config.stock,
            start_date=self.pred_config.start_date,
            end_date=self.pred_config.end_date,
        )

        self.feat_cols = []
        for col, enabled in self.pred_config.stock_data_options.items():
            if enabled:
                self.feat_cols.append(col)
        if self.pred_config.vader_on or self.pred_config.finbert_on:
            self.feat_cols.append("sentiment")
        self.label_col = "close"

    def run(self):
        """
        Main function to run stock prediction with sentiment analysis
        """
        # Extract and preprocess historical price and news sentiment data
        input_data = self.extract_sentiment_and_prices()

        # Run prediction and generate results
        self.run_lstm(input_data)

        # Option to download data if user wishes to review them
        self.download_options()

    def extract_sentiment_and_prices(self):
        """
        Extract historical prices and news sentiment data for stock
        """
        # Extract and save historical news data
        archive_df = FinnHubNews(ticker=self.pred_config.stock, 
                                 start_date=self.pred_config.start_date, 
                                 end_date=self.pred_config.end_date, 
                                 always_extract=False).get_news()
        finnhub_df = InternetArchiveNews(ticker=self.pred_config.stock, 
                                         start_date=self.pred_config.start_date, 
                                         end_date=self.pred_config.end_date, 
                                         always_extract=False).get_news()
        sentiment_df = pd.concat([archive_df, finnhub_df], ignore_index=True)
        print(sentiment_df)
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        sentiment_df.sort_values(by="date", ignore_index=True, inplace=True)

        # Extract and save historical stock price data
        stock_data_df = get_stock_data_v2(
            ticker=self.pred_config.stock, 
            start_date=self.pred_config.start_date,
            end_date=self.pred_config.end_date, 
            export_data=True
        )

        # Gather sentiment scores
        total_sentiment_scores = pd.DataFrame()
        if self.pred_config.vader_on:
            # Get VADER sentiment scores
            vader_sentiments_df = VaderSentiment(text_df=sentiment_df).generate()
            total_sentiment_scores = pd.concat(
                [total_sentiment_scores, vader_sentiments_df], 
                ignore_index=True
            )
            print(f"VADER:\n{vader_sentiments_df}")

        if self.pred_config.finbert_on:
            # Get FinBERT sentiment scores
            finbert_sentiments_df = FinBertSentiment(text_df=sentiment_df).generate()
            total_sentiment_scores = pd.concat(
                [total_sentiment_scores, finbert_sentiments_df], 
                ignore_index=True
            )
            print(f"FINBERT:\n{finbert_sentiments_df}")

        if self.pred_config.textblob_on:
            # Generate TextBlob sentiment scores
            textblob_sentiments_df = TextBlobSentiment(text_df=sentiment_df).generate()
            total_sentiment_scores = pd.concat(
                [total_sentiment_scores, textblob_sentiments_df], 
                ignore_index=True
            )
            print(f"TEXTBLOB:\n{textblob_sentiments_df}")
        
        if self.pred_config.flair_on:
            # Generate Flair sentiment scores
            flair_sentiments_df = FlairSentiment(text_df=sentiment_df).generate()
            total_sentiment_scores = pd.concat(
                [total_sentiment_scores, flair_sentiments_df], 
                ignore_index=True
            )
            print(f"FLAIR:\n{flair_sentiments_df}")
        
        print(f"total_sentiment_scores:\n{total_sentiment_scores}")
        
        # Group and average sentiment scores of news per day
        sentiment_df = pd.DataFrame()
        if not total_sentiment_scores.empty:
            sentiment_df = average_sentiment_per_date(sentiment_scores=total_sentiment_scores)
        
            # Remove sentiment string column
            sentiment_df.drop(['sentiment'], axis=1, inplace=True) 

            print("\nSENTIMENT_DATA\n")
            print(sentiment_df)

        print("\nPRICE_DATA\n")
        print(stock_data_df)

        # Merge sentiment data with historical stock data into one dataframe
        if not sentiment_df.empty:
            data = pd.merge(stock_data_df, sentiment_df, on="date", how="left")
            print(f"\nMERGED DATA:\n\n{data}")

            # Fill in missing gaps in the merged data (trying out interpolations and bfill)
            data = data.interpolate(method='linear')
            data = data.fillna(method='bfill')
            print(f"\nMERGED DATA (after interpolation):\n\n{data}")

            # Rename "compound" to "sentiment"
            data.rename(columns={"compound": "sentiment"}, inplace=True)
        else:
            # If no sentiment data, then only using historical stock data
            data = stock_data_df

        # Return input data
        return data

    def run_lstm(self, df):
        """
        Run LSTM to make predictions
        Source: https://github.com/vb100/multivariate-lstm/blob/master/LSTM_model_stocks.ipynb
        :param df: input dataframe in raw form
        """
        # Extract dates (will be used in visualization)
        datelist_train = list(df['date'])

        print('Training set shape == {}\n{}'.format(df.shape, df))
        print('All timestamps == {}'.format(len(datelist_train)))
        print('Featured selected: {}'.format(self.feat_cols))

        # Set date column as index
        df.set_index('date', inplace=True)

        # Rescale data
        sc = StandardScaler()
        training_set_scaled = sc.fit_transform(df)

        sc_predict = StandardScaler()
        sc_predict.fit_transform(df[[self.label_col]])

        # Create X_train and y_train
        X_train = []
        y_train = []

        for i in range(self.pred_config.n_past, len(training_set_scaled) - self.pred_config.n_future +1):
            X_train.append(training_set_scaled[i - self.pred_config.n_past:i, 0:df.shape[1] - 1])
            y_train.append(training_set_scaled[i + self.pred_config.n_future - 1:i + self.pred_config.n_future, 0])

        X_train, y_train = np.array(X_train), np.array(y_train)

        print('X_train shape == {}.'.format(X_train.shape))
        print('y_train shape == {}.'.format(y_train.shape))

        st.write("LSTM Model Building")

        # Initializing the Neural Network based on LSTM
        model = Sequential()
        model.add(LSTM(units=self.pred_config.n_units, return_sequences=True, input_shape=(self.pred_config.n_past, df.shape[1]-1)))
        model.add(Dropout(self.pred_config.dropout))

        for _ in range(self.pred_config.n_layers):
            model.add(LSTM(units=self.pred_config.n_units, return_sequences=True))
            model.add(Dropout(self.pred_config.dropout))
        
        model.add(LSTM(units=self.pred_config.n_units))
        model.add(Dropout(self.pred_config.dropout))
        model.add(Dense(units=1))

        # Compiling the Neural Network
        model.compile(optimizer = Adam(learning_rate=self.pred_config.learning_rate), 
                      loss='mean_squared_error')
        model.summary(print_fn=lambda x: st.text(x))

        # EarlyStopping - Stop training when a monitored metric has stopped improving.
        es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)

        # ReduceLROnPlateau - Reduce learning rate when a metric has stopped improving.
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)    
        mcp = ModelCheckpoint(filepath=self.file_location.get_saved_model_name(), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)

        # Create tensorboard for logging
        tb = TensorBoard('logs')

        # Perform prediction
        history = model.fit(X_train, y_train, 
                            shuffle=True, 
                            epochs=self.pred_config.n_epochs, 
                            callbacks=[es, rlr, mcp, tb], 
                            validation_split=0.2, 
                            verbose=1, 
                            batch_size=self.pred_config.batch_size)
        hist_df = pd.DataFrame(history.history)
        with open(self.file_location.get_hist_train_name(), mode='w') as f:
            hist_df.to_csv(f)
        st.write(f"Saved LSTM Model Training History:")
        st.write(hist_df)

        # Generate list of sequence of days for predictions
        datelist_future = pd.date_range(datelist_train[-1], periods=self.pred_config.n_future, freq='1d').tolist()
        print("datelist_future:", datelist_future)

        # Convert Pandas Timestamp to Datetime object (for transformation) --> FUTURE
        datelist_future_ = []
        for this_timestamp in datelist_future:
            datelist_future_.append(this_timestamp.date())
        
        # Perform predictions
        predictions_future = model.predict(X_train[-self.pred_config.n_future:])
        predictions_train = model.predict(X_train[self.pred_config.n_past:])

        # Inverse the predictions to original measurements

        # Special function: convert <datetime.date> to <Timestamp>
        def datetime_to_timestamp(x):
            '''
                x : a given datetime value (datetime.date)
            '''
            return datetime.strptime(x.strftime('%Y%m%d'), '%Y%m%d')

        y_pred_future = sc_predict.inverse_transform(predictions_future)
        y_pred_train = sc_predict.inverse_transform(predictions_train)

        PREDICTIONS_FUTURE = pd.DataFrame(y_pred_future, columns=[self.label_col]).set_index(pd.Series(datelist_future))
        PREDICTION_TRAIN = pd.DataFrame(y_pred_train, columns=[self.label_col]).set_index(pd.Series(datelist_train[2 * self.pred_config.n_past + self.pred_config.n_future -1:]))

        # Convert <datetime.date> to <Timestamp> for PREDCITION_TRAIN
        PREDICTION_TRAIN.index = PREDICTION_TRAIN.index.to_series().apply(datetime_to_timestamp)

        # Set plot size 
        rcParams['figure.figsize'] = 14, 5

        # Plot parameters
        # START_DATE_FOR_PLOTTING = self.start_date.strftime("%Y-%m-%d")
        START_DATE_FOR_PLOTTING = self.start_date_for_plotting

        # Calculate scoring
        y_pred = PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:][self.label_col].to_frame()
        y_test = df.loc[START_DATE_FOR_PLOTTING:][self.label_col].to_frame()
        y_pred.index.name = 'date'
        y_pred.rename(columns={"close": "y_pred_close"}, inplace=True)
        y_test.rename(columns={"close": "y_test_close"}, inplace=True)
        together = pd.merge(y_pred, y_test, left_index=True, right_index=True, how='inner')
        self.generate_scores(
            y_pred=together["y_pred_close"].to_numpy(),
            y_test=together["y_test_close"].to_numpy(),
        )

        st.write(f"Visualize Predictions for {self.pred_config.stock}")

        fig = plt.figure()
        plt.plot(PREDICTIONS_FUTURE.index, 
                 PREDICTIONS_FUTURE[self.label_col], 
                 color='r', 
                 label='Predicted Stock Price')
        plt.plot(PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:].index, 
                 PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:][self.label_col], 
                 color='orange', 
                 label='Training Predictions')
        plt.plot(df.loc[START_DATE_FOR_PLOTTING:].index, 
                 df.loc[START_DATE_FOR_PLOTTING:][self.label_col], 
                 color='b', 
                 label='Actual Stock Price')
        plt.axvline(x = min(PREDICTIONS_FUTURE.index), 
                    color='green', 
                    linewidth=2, 
                    linestyle='--')
        plt.grid(which='major', 
                 color='#cccccc', 
                 alpha=0.5)
        plt.legend(shadow=True)
        plt.title(f'Predictions and Actual Stock Prices for {self.pred_config.stock}', family='Arial', fontsize=12)
        plt.xlabel('Timeline', family='Arial', fontsize=10)
        plt.ylabel('Stock Price Value', family='Arial', fontsize=10)
        plt.xticks(rotation=45, fontsize=8)
        
        fig_html = mpld3.fig_to_html(fig)
        components.html(fig_html, height=550)

        # Save image for download
        self.buf = io.BytesIO()
        plt.savefig(self.buf, format="png")
        self.buf.seek(0)

    def download_options(self):
        """
        Download buttons to download:
        - News data
        - Historical Stock Price data
        - Graph of stock price prediction
        """
        def convert_df(df):
           return df.to_csv(index=False).encode('utf-8')

        dl_finnhub_sentiment, dl_archive_sentiment, dl_hist_stock, dl_graph, dl_hist_training = st.columns([1,1,1,1,1])
        with dl_finnhub_sentiment:
            finnhub_filepath = self.file_location.get_finnhub_news_name()
            finnhub_data = pd.read_csv(finnhub_filepath)
            st.download_button(
                label="Download FinnHub Sentiment News Data",
                data=convert_df(finnhub_data),
                file_name=os.path.basename(finnhub_filepath),
            )
        with dl_archive_sentiment:
            archive_filepath = self.file_location.get_archive_news_name()
            archive_data = pd.read_csv(archive_filepath)
            st.download_button(
                label="Download Internet Archive Sentiment News Data",
                data=convert_df(archive_data),
                file_name=os.path.basename(archive_filepath),
            )
        with dl_hist_stock:
            hist_stock_filepath = self.file_location.get_hist_stock_data_name()
            hist_stock_data = pd.read_csv(hist_stock_filepath)
            st.download_button(
                label="Download Historical Stock Data",
                data=convert_df(hist_stock_data),
                file_name=os.path.basename(hist_stock_filepath),
            )
        with dl_graph:
            graph_filepath = self.file_location.get_graph_name()
            st.download_button(
                label="Download Prediction Graph",
                data=self.buf,
                file_name=os.path.basename(graph_filepath),
                mime="image/png"
            )
        with dl_hist_training:
            hist_train_filepath = self.file_location.get_hist_stock_data_name()
            hist_train_data = pd.read_csv(hist_train_filepath)
            st.download_button(
                label="Download Training History",
                data=convert_df(hist_train_data),
                file_name=os.path.basename(hist_train_filepath),
            )

    def calculate_rmse(self, y_true, y_pred):
        """
        Calculate the Root Mean Squared Error (RMSE)
        """
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        return rmse

    def calculate_mape(self, y_true, y_pred):
        """
        Calculate the Mean Absolute Percentage Error (MAPE) %
        """
        y_pred, y_true = np.array(y_pred), np.array(y_true)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return mape

    def generate_scores(self, y_test, y_pred):
        """
        Generate evaluation scores
        :param y_test: true values
        :param y_pred: predicted values
        """
        st.write(f"Evaluation ({len(y_test)} samples):")
        
        print(f"y_test (Size: {len(y_test)}):\n{y_test}")
        print(f"y_pred (Size: {len(y_pred)}):\n{y_pred}")

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = self.calculate_rmse(y_test, y_pred)
        mape = self.calculate_mape(y_test, y_pred)
        
        measurements_df = pd.DataFrame({
            "Measurement": ["MAE", "MSE", "R2", "RMSE", "MAPE"], 
            "Score": [mae, mse, r2, rmse, mape],
        })
        st.write(measurements_df)


def get_stock_tickers():
        """
        Extract list of stock tickers that exist for data extraction.
        """
        results = pd.read_json(LOCATIONS.TICKER_DATA_LOCATION).T
        return results["ticker"]


def run_app():
    '''
    Run application for stock price prediction, using streamlit
    '''
    st.title("Stock Price Predictor using Sentiment & Historical Data")

    # Get dates
    start_date_col, end_date_col = st.columns([1, 1])
    with start_date_col:
        last_year = datetime.now().year - 1
        last_year_date = date(last_year, 1, 1)
        start_date = st.date_input("Enter start date", value=last_year_date)
    with end_date_col:
        end_date = st.date_input("Enter end date", value="today")

    # Get stock ticker
    tickers = get_stock_tickers()
    ticker_options = st.multiselect(
        label="Select stock ticker(s)",
        options=tickers.unique(),
    )

    # Options to enable certain historical stock data or sentiment analysis scoring
    historical_col, sentiment_col = st.columns([1, 1])
    with historical_col:
        st.write("Enable historical stock price metrics:")
        hist_left_col, hist_mid_col, hist_right_col = st.columns([1, 1, 1])
        with hist_left_col:
            open_tog_on = st.toggle("Open", value=True)
            close_tog_on = st.toggle("Close", value=True)
        with hist_mid_col:
            high_tog_on = st.toggle("High", value=True)
            low_tog_on = st.toggle("Low", value=True)
        with hist_right_col:
            vol_tog_on = st.toggle("Volume", value=True)
    with sentiment_col:
        st.write("Enable news sentiment evaluators:")
        sentiment_left, sentiment_right = st.columns([1, 1])
        with sentiment_left:
            vader_on = st.toggle("VADER", value=True)
            finbert_on = st.toggle("FinBERT", value=True)
        with sentiment_right:
            textblob_on = st.toggle("TextBlob", value=True)
            flair_on = st.toggle("Flair", value=True)

    # Range of what number of days to look back and forward for prediction
    st.write("Set Training and Prediction Days:")
    past_col, future_col = st.columns([1, 1])
    with past_col:
        n_past = st.number_input(
            "Insert number of days to look in the past for training", 
            value=90, 
            min_value=0,
            format="%d",
            placeholder="Type a number..."
        )
    with future_col:
        n_future = st.number_input(
            "Insert number of days to predict in the future", 
            value=30, 
            min_value=0,
            format="%d",
            placeholder="Type a number..."
        )

    # Training configurations (epochs, batch size, number of layers, number of units, learning rate)
    st.write("Set LSTM hyperparameters:")
    lstm_col1, lstm_col2 = st.columns([1, 1])
    with lstm_col1:
        n_epochs = st.number_input(
            "Insert number of epochs", 
            value=50, 
            min_value=1,
            format="%d",
            placeholder="Type a number..."
        )
        batch_size = st.number_input(
            "Insert batch size", 
            value=32, 
            min_value=1,
            format="%d",
            placeholder="Type a number..."
        )
        learning_rate = st.number_input(
            "Insert learning rate", 
            value=0.001, 
            min_value=0.00000001,
            format="%.8f",
            placeholder="Type a decimal rate..."
        )
    with lstm_col2:
        n_layers = st.number_input(
            "Insert number of layers in LSTM model (2 minimum)", 
            value=4, 
            min_value=2,
            max_value=10,
            format="%d",
            placeholder="Type a number..."
        )
        n_units = st.number_input(
            "Insert number of units for each layer in LSTM model", 
            value=96, 
            min_value=1,
            format="%d",
            placeholder="Type a number..."
        )
        dropout = st.number_input(
            "Insert dropout rate (0.0-1.0) for each layer in LSTM model", 
            value=0.2, 
            min_value=0.0,
            max_value=1.0,
            format="%.f",
            placeholder="Type a decimal rate..."
        )

    # Run prediction
    if st.button("Predict"):
        # For every stock listed in ticker options,
        for stock in ticker_options:
            try: 
                st.write(f"Running Predictions for **{stock}**")

                # Set up configurations for prediction
                pred_config = PredictionConfig(
                    stock=stock, 
                    start_date=start_date,
                    end_date=end_date,
                    open=open_tog_on,
                    close=close_tog_on,
                    high=high_tog_on,
                    low=low_tog_on,
                    vol=vol_tog_on,
                    vader=vader_on,
                    finbert=finbert_on,
                    textblob=textblob_on,
                    flair=flair_on,
                    n_past=n_past,
                    n_future=n_future,
                    n_units=n_units,
                    n_epochs=n_epochs,
                    n_layers=n_layers,
                    batch_size=batch_size,
                    dropout_rate=dropout,
                    learning_rate=learning_rate,
                )

                # Run prediction
                predictor = StockPricePredictionSentiment(config=pred_config)
                predictor.run()
            except Exception as e:
                st.write(f"Cannot run prediction for stock {stock}: {e}")
                raise Exception(e)


if __name__=="__main__": 
    run_app()