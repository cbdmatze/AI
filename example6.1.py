import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Read teh text from the file
file_path = "NLP/data/preprocessed_romeo_and_juliet.txt"
with open(file_path, "r") as f:
    text = f.read()

# Tokenize the text into sentences
sentences = nltk.sent_tokenize(text)

# Tokenize each sentence into words and tag POS
noun_verb_pairs = []
for sentence in sentences:
    tokens = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)

    # Find all instances of nouns followed by verbs
    for i in range(len(pos_tags) - 1):
        if pos_tags[i][1].startswith("NN") and pos_tags[i + 1][1].startswith("VB"):
            noun_verb_pairs.append((pos_tags[i][0], pos_tags[i + 1][0]))

# Print the noun-verb pairs
print("Noun-verb pairs:")
for pair in noun_verb_pairs:
    print(pair)
