from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from flair.models import TextClassifier
from flair.data import Sentence

import torch

# Load pre-trained sentiment model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
flair_classifier = TextClassifier.load('sentiment')
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)

from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
import re
import streamlit as st


class SentimentAnalysis:
    def __init__(self, text_df: pd.DataFrame):
        """
        :param text_df: dataframe of news/social media text and corresponding dates
        Format columns must be: ["date", "title", "summary"]
        """
        self.text_df = text_df

    def get_name(self):
        """
        Get type of Sentiment
        """
        pass

    def get_score(self, text: str):
        """
        Generate sentiment score with given text
        :param text: input string text to generate  
        """
        pass

    def generate(self):
        """
        Generate sentiment corresponding to list of texts
        :param text_df: dataframe of news/social media text and corresponding dates
        :return Pandas Dataframe of sentiment scores with corresponding text
        """
        # Create sentiment score dict
        sentiment_score = {
            "date": [],
            "compound": [],
        }

        # Generate sentiment scores for every text
        progress_bar = st.progress(0, f"Generating {self.get_name()} sentiment scores...")
        i = 0
        total_searches = len(self.text_df)
        prg = 0

        for _, row in tqdm(self.text_df.iterrows()):
            # Calculate progress bar
            i += 1
            prg = i/total_searches
            progress_bar.progress(prg, f"Generating {i}/{total_searches} ({format(prg * 100, ".2f")}%) {self.get_name()} sentiment scores (This may take a while; Do not refresh page)")

            # Extract title or summary as text
            if pd.isna(row["summary"]):
                text = row["title"]
            else:
                text = row["summary"]

            # Replace newlines with spaces
            text = text.replace('\n', ' ')

            # Replace multiple spaces with a single space
            text = re.sub(' +', ' ', text)

            # Generate compound score
            compound_score = self.get_score(text)

            # Append date and score data
            sentiment_score["date"].append(row["date"])
            sentiment_score["compound"].append(compound_score)
        
        progress_bar.progress(prg, f"{self.get_name()} sentiment scores complete!")

        # Convert to pandas dataframe
        sentiment_score = pd.DataFrame(sentiment_score)

        # Return scores
        return sentiment_score


class VaderSentiment(SentimentAnalysis):
    def __init__(self, text_df):
        super().__init__(text_df)
        self.sid_obj = SentimentIntensityAnalyzer()

    def get_name(self):
        return "VADER"

    def get_score(self, text):
        """
        Inheritence function to generate score using specified 
        """
        sentiment_dict = self.sid_obj.polarity_scores(text)
        return sentiment_dict['compound']
    

class FinBertSentiment(SentimentAnalysis):
    def __init__(self, text_df):
        super().__init__(text_df)

    def get_name(self):
        return "FinBERT"

    def get_score(self, text):
        """
        Inheritence function to generate score using specified 
        """
        # Shorten text if exceeds limit
        if len(text) > 512:
            text = text[:512]
        
        # Get predictions
        encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt") 
        output = model(**encoded_input) 

        # Get sentiment probabilities
        predictions = torch.nn.functional.softmax(output.logits, dim=-1)

        # Extract sentiment labels and scores
        positive_score = predictions[0][0].item()
        negative_score = predictions[0][1].item()
        # neutral_score = predictions[0][2].item()

        # Return compound score
        compound_score = positive_score - negative_score
        return compound_score
    

class TextBlobSentiment(SentimentAnalysis):
    def __init__(self, text_df):
        super().__init__(text_df)

    def get_name(self):
        return "TextBlob"

    def get_score(self, text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity


class FlairSentiment(SentimentAnalysis):
    def __init__(self, text_df):
        super().__init__(text_df)

    def get_name(self):
        return "Flair"

    def get_score(self, text):
        """
        Analyze the sentiment of text using Flair and rescale the score to -1 to 1.
        Code Source: ChatGPT
        Args:
            text (str): The input text.

        Returns:
            float: Rescaled sentiment score (-1 to 1).
        """
        # Prepare the text sentence
        sentence = Sentence(text)

        # Perform sentiment analysis
        flair_classifier.predict(sentence)

        # Extract the label and score
        if len(sentence.labels) > 0:
            label = sentence.labels[0].value  # SENTIMENT_POSITIVE or SENTIMENT_NEGATIVE
            score = sentence.labels[0].score  # Confidence score (0.0 to 1.0)

            # Rescale score: positive stays positive, negative becomes negative
            if label == "POSITIVE":
                return score  # Score between 0 to 1
            elif label == "NEGATIVE":
                return -score  # Score between -1 to 0
            else:
                raise ValueError("Unexpected sentiment label: {}".format(label))
        else:
            # If not found, then return 0 for neutral
            return 0
    

def average_sentiment_per_date(sentiment_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Condense dataframe to have single rows for every date, and them 
    :param sentiment_scores: sentiment score dataframe
    """
    # Filter out neutral scores
    sentiment_scores = sentiment_scores[
        (sentiment_scores["compound"] >= 0.05) | (sentiment_scores["compound"] <= -0.05)
    ]

    # Reformat date column to only show yyyy-mm-dd (remove time)
    sentiment_scores['date'] = pd.to_datetime(sentiment_scores['date'])

    # For every group, get average setiment scores
    grouped_sentiments = sentiment_scores.groupby('date')["compound"].mean().reset_index()
    
    # Sort dataframe by date descending order, and group them by date
    grouped_sentiments = grouped_sentiments.sort_values(by='date')

    # Add sentiment
    grouped_sentiments['sentiment'] = np.where(
        (grouped_sentiments["compound"] >= 0.05) | (grouped_sentiments["compound"] <= -0.05), 
        np.where(grouped_sentiments["compound"] >= 0.05, 'positive', 'negative'), 
        'neutral'
    )

    # Return condensed dataframe
    return grouped_sentiments