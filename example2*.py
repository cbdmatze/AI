# pip install spacy
# python -m spacy download en_core_web_sm

import spacy
from collections import Counter
from itertools import islice

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Read the text from the file
file_path = "data/preprocessed_text.txt"
with open(file_path, "r") as f:
    text = f.read()

# Process the text with spaCy
doc = nlp(text)

# Tokenize the text into words
tokens = [token.text for token in doc]

# Function to generate n-grams
def generate_ngrams(tokens, n):
    return zip(*[islice(tokens, i, None) for i in range(n)])

# Generate Unigrams (1-grams)
unigrams = list(generate_ngrams(tokens, 1))
print("Unigrams:")
print(unigrams)

# Generate Bigrams (2-grams)
bigrams = list(generate_ngrams(tokens, 2))
print("\nBigrams:")
print(bigrams)

# Generate Trigrams (3-grams)
trigrams = list(generate_ngrams(tokens, 3))
print("\nTrigrams:")
print(trigrams)

# Count Frequency of each n-gram
unigram_freq = Counter(unigrams)
bigram_freq = Counter(bigrams)
trigram_freq = Counter(trigrams)

# Print Frequencies
print("\nUnigram Frequencies:")
print(unigram_freq)

print("\nBigram Frequencies:")
print(bigram_freq)

print("\nTrigram Frequencies:")
print(trigram_freq)

# Find and print the most common n-grams
most_common_unigram = unigram_freq.most_common(1)
most_common_bigram = bigram_freq.most_common(1)
most_common_trigram = trigram_freq.most_common(1)

print("\nMost Common Unigram and its Frequency:")
print(most_common_unigram)

print("\nMost Common Bigram and its Frequency:")
print(most_common_bigram)

print("\nMost Common Trigram and its Frequency:")
print(most_common_trigram)
