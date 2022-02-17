from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def label(text):

    def getSubjectivity(text):
        return TextBlob(text).sentiment.subjectivity

    def getPolarity(text):
        return TextBlob(text).sentiment.polarity

    def getSIA(text):
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(text)
        return sentiment

    text = 'fml'

    sia = getSIA(text)

    # Variable Subjectivity
    Subjectivity = getSubjectivity(text)

    # Variable Polarity
    Polarity = getPolarity(text)

    # Variable Compound
    Compound = sia['compound']

    return Subjectivity, Polarity, Compound
