import spacy
from nltk.stem import PorterStemmer
import re

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = "My dog is named LeBron and enjoys chewing sticks!"

# Lowercasing
text_lower = text.lower()
print("Lowercased text:", text_lower)

# Removing punctuation
text_no_punct = re.sub(r'[^\w\s]', '', text_lower)
print("Text without punctuation:", text_no_punct)

# Process the text with spaCy
doc = nlp(text_no_punct)

# Tokenization
words = [token.text for token in doc]
print("Tokenized words:", words)

# Removing stop words
words_no_stop = [token.text for token in doc if not token.is_stop]
print("Words without stop words:", words_no_stop)

# Stemming using NLTK's PorterStemmer
ps = PorterStemmer()
words_stemmed = [ps.stem(word) for word in words_no_stop]
print("Stemmed words:", words_stemmwed)

# Lemmatizing using spaCy
words_lemmatized = [token.lemma_ for token in doc if not token.is_stop]
print("Lemmatized words:", words_lemmatized)
