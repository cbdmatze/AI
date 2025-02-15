text = "I absolutely love this phone! The battery life is amazing."
positive_words = ["love", "amazing", "great", "awesome"]

# Convert the text to lowercase and split into words
words = text.lower().split()

# Check if each word is in the positive words list
positive_matches = [word in positive_words for word in words]

# Sum up the number of matches
score = sum(positive_matches)

if score > 0:
    print("Positive")
else:
    print("Negative or Neutral")
   