import streamlit as st
import pandas as pd
import tensorflow as tf
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def user_report():
    input_text = st.text_area('Tweets', 'Enter your tweets here')
    review_data = str(input_text)

    return review_data


user_input = user_report()
#st.write(user_input)
#review_data = user_input.to_string()

def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    return TextBlob(text).sentiment.polarity

def getSIA(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment

sia = getSIA(user_input)

#pd.set_option('precision', 4)
# st.write(comp)
#st.write(sia['compound'])
#Variable Subjectivity
Subjectivity = getSubjectivity(user_input)
# st.write(subjectivity)
#Variable Polarity
Polarity = getPolarity(user_input)
# st.write(polarity)
#Variable Compound
Compound= sia['compound']

