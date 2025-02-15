import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re


# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample text for performing NLP operations
text = "My dog is named LeBron and enjoys chewing sticks!"

# Lowercasing
text_lower = text.lower()
print("Lowercased text:", text_lower)

# Removing punctuation
text_no_punct = re.sub(r'[^\w\s]', '', text_lower)
print("Text without punctuation:", text_no_punct)

# Tokenization
words = nltk.word_tokenize(text_no_punct)
print("Tokenized words:", words)

# Removing stop words
stop_words = set(stopwords.words('english'))
words_no_stop = [word for word in words if word not in stop_words]
print("Words without stop words:", words_no_stop)

# Stemming
ps = PorterStemmer()
words_stemmed = [ps.stem(word) for word in words_no_stop]
print("Stemmed words:", words_stemmed)

# Lemmatization
lemmatizer = WordNetLemmatizer()
words_lemmatized = [lemmatizer.lemmatize(word) for word in words_no_stop]
print("Lemmatized words:", words_lemmatized)
