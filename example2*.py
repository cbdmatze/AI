import nltk
from nltk.util import ngrams
from collections import Counter

# Download necessary NLTK data
nltk.download('punkt')

# Sample text data
file_path = "NLP/data/preprocessed_romeo_and_juliet.txt"
with open(file_path, 'r') as f:
    text = f.read()

# Tokenize the text into words
tokens = nltk.word_tokenize(text)

# Generate Unigrams (1-grams
unigrams = list(ngrams(tokens, 1))
print("Unigrams:")
print(unigrams)

# Generate Bigrams (2-grams)
bigrams = list(ngrams(tokens, 2))
print("\nBigrams:")
print(bigrams)

# Generate Trigrams (3-grams)
trigrams = list(ngrams(tokens, 3))
print("\nTrigrams:")
print(trigrams)

# Count Frequency of each n-gram (for demonstration)
unigram_freq = Counter(unigrams)
bigram_freq = Counter(bigrams)
trigram_freq = Counter(trigrams)

# Print Frequencies (optional)
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

print("\nMost Common Unigram and its frequency:")
print(most_common_unigram)

print("\nMost Common Bigram and its frequency:")
print(most_common_bigram)

print("\nMost Common Trigram and its frequency:")
print(most_common_trigram)
