"""
Main code to run app
"""

import pandas as pd
import streamlit as st

from stock_price_prediction import PredictionConfig, StockPricePredictionSentiment
from datetime import datetime, date
from names import LOCATIONS


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
    tickers = pd.read_json(f"{LOCATIONS.TICKER_DATA_LOCATION}/tickers.json").T["ticker"]
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