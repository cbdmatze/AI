import nltk

nltk.download('averaged_perceptron_tagger', download_dir='/tmp')
text = "The quick brown fox jumps over the lazy dog"
tokens = nltk.word_tokenize(text)
pos_tags = nltk.pos_tag(tokens)
print(pos_tags)
