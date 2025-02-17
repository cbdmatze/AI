# pip install spacy
# python -m spacy download en_core_web_sm
import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Read the text from the file
file_path = "NLP/data/preprocessed_romeo_and_juliet.txt"
with open(file_path, "r") as f:
    text = f.read()

# Process the text with spaCy
doc = nlp(text)

# Extract named entities with the DATE label
for ent in doc.ents:
    if ent.label_ == "DATE":
        print(ent.text, ent.label_)
