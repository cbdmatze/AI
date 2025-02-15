import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# Download necessary NLTK data
nltk.download('punkt')

# Sample text data
text = [
    "The children went to school",
"The teacher taught at the school",
    "The school library is full of books",
"The cat was seen near the school",
]

# Tokenize the sentences into words
tokenized_text = [word_tokenize(sentence.lower()) for sentence in text]

# Train a Word2Vec model on the tokenized text
model = Word2Vec(tokenized_text, vector_size=100, window=5, min_count=1, sg=0)

# Get the word embeddings for a specific word
school_vector = model.wv['school']

# Print the word embedding for 'school'
print("Word Embedding for 'school':")
print(school_vector)

# Find words most similar to 'school'
similar_words = model.wv.most_similar('school', topn=3)
print("\nWords most similar to 'school':")
print(similar_words)
