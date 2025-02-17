# pip install spacy
# python -m spacy download en_core_web_sm
import spacy

# Load the spaCy model 
nlp = spacy.load("en_core_web_sm")

# Sample text data
text = "The quick brown fox jumps over the lazy dog."

# Process the text with spaCy
doc = nlp(text)

# Extract and pring POS tags
pos_tags = [(token.text, token.pos_) for token in doc]
print(pos_tags)
