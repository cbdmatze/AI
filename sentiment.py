def sentiment(text):
    # This is a simplified version of the actual implementation
    # The actual implementation involves a lot of preprocessing and analysis
    polarity = 0.0
    subjectivity = 0.0

    # Example logic to calculate polarity and subjectivity
    words = text.split()
    for word in words:
        if word in positive_words:
            polarity += 1.0
            subjectivity += 0.5
        elif word in negative_words:
            polarity -= 1.0
            subjectivity += 0.5

    # Normalize the values
    polarity = polarity / len(words)
    subjectivity = subjectivity / len(words)

    return polarity, subjectivity