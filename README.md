# LSTM_Stock_Prediction_Sentiment_Analysis
Python project to predict future stock prices based on historical prices and sentiment analysis of historical news data.

- Historical prices extracted from yfinance, using `Open`, `Close`, `High`, `Low` and `Volume`
- Historical news data extracted from `FinnHub` and `Internet Archive`
- Sentiment Analysis is scored using `VADER`, `FinBERT`, `TextBlob` and `Flair`
- Prediction is run with customizable LSTM
- App is built on `Streamlit`

See `requirements.txt` for all libraries and versions needed.

To run app:
- Deployed:
- Code: 
    1. Download repo and requirements
    2. Inside `lstm_stock_prediction_sentiment_analysis` main folder, run `streamlit run code/run.py`