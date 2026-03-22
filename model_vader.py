# model_vader.py
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

# VADER knows words like "amazing" = positive, "terrible" = negative
# It was built specifically for social media text
sia = SentimentIntensityAnalyzer()

def predict_vader(text):
    scores = sia.polarity_scores(text)
    # scores looks like: {'neg': 0.0, 'neu': 0.3, 'pos': 0.7, 'compound': 0.8}
    
    compound = scores['compound']  # overall score from -1 to +1

    if compound >= 0.05:
        return "Positive", round(compound * 100, 1)
    elif compound <= -0.05:
        return "Negative", round(abs(compound) * 100, 1)
    else:
        return "Neutral", 50.0

# Test it
reviews = [
    "This is absolutely fantastic! Best product ever!",
    "Terrible quality. Complete waste of money.",
    "It arrived on time. Does what it says."
]

for r in reviews:
    sentiment, confidence = predict_vader(r)
    print(f"Review: {r[:40]}...")
    print(f"Sentiment: {sentiment} ({confidence}% confident)\n")