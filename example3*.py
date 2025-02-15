# pip install spacy
# python -m spacy download en_core_web_md

import spacy

# Load the spaCy model with vectors
nlp = spacy.load("en_core_web_md")

# Sample text data
text = [
    "The children went to school",
    "The teacher taught at the school",
    "The school library is full of books",
    "The cat was seen near the school",
]

# Process thge text with spaCy
docs = [nlp(sentence.lower()) for sentence in text]

# Tokenize the sentences into words
tokenized_text = [[token.text for token in doc] for doc in docs]

# Get the word embeddings for a specific word
school_vector = nlp("school").vector

# Print the word embedding for 'school'
print("Word Embedding for 'school':")
print(school_vector)

# Find words most similar to 'school'
similar_words = nlp.vocab.vectors.most_similar(school_vector.reshape(1, school_vector.shape[0]), n=3)
similar_words = [nlp.vocab.strings[w] for w in similar_words[0][0]]

print("\nWords most similar to 'school':")
print(similar_words)
