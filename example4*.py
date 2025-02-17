# pip install spacy textblob
# python -m spacy download en_core_web_sm
import spacy
from textblob import TextBlob

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = "I ablolutely love this phone! The battery life is amazing."

# Process the text with spaCy
doc = nlp(text)

# Convert the spaCy doc to a string and create a TextBlob object
blob = TextBlob(doc.text)

# Get the sentiment polarity
sentiment = blob.sentiment.polarity

# Determine if the sentiment is positive, negative, or neutral
if sentiment > 0:
    print("Positive")
elif sentiment < 0:
    print("Negative")
else:
    print("Neutral")
