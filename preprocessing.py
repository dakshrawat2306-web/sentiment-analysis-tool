# preprocessing.py
import nltk
import re
nltk.download('stopwords')
nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Step 1: Lowercase everything
    text = text.lower()

    # Step 2: Remove URLs (e.g. http://...)
    text = re.sub(r'http\S+', '', text)

    # Step 3: Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Step 4: Tokenize — split into individual words
    # "I love this phone" → ["I", "love", "this", "phone"]
    tokens = word_tokenize(text)

    # Step 5: Remove stopwords — words like "the", "is", "a"
    # These add noise but carry no sentiment meaning
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)