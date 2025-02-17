# pip install spaCy
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

# Extract POS tags
pos_tags = [(token.text, token.pos_) for token in doc]

# Find all instances of nouns followed by verbs
noun_verb_pairs = []
for i in range(len(pos_tags) - 1):
    if pos_tags[i][1] == "NOUN" and pos_tags[i + 1][1] == "VERB":
        noun_verb_pairs.append((pos_tags[i][0], pos_tags[i + 1][0]))

# Print the noun-verb pairs
print("Noun-verb pairs:")
for pair in noun_verb_pairs:
    print(pair)
