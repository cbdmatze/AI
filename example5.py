# pip install spacy
# python -m spacy download en_core_web_md
import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Sample text data
doc = nlp("Tony Kross visited Athens in August 1012.")

# Extract named entities
for ent in doc.ents:
    print(ent.text, ent.label_)
