from collections import namedtuple
from pattern.en import sentiment as pattern_sentiment

Sentiment = namedtuple('Sentiment', ['polarity', 'subjectivity'])

class PatternAnalyzer:
    """
    A class to represent a PatternAnalyzer object.
    
    Methods
    -------
    analyze_sentiment(text)
        Analyzes the sentiment of the given text.
    
    Examples
    --------
    >>> pa = PatternAnalyzer()
    >>> text = "I absolutely love this phone! The battery life is amazing."
    >>> pa.analyze_sentiment(text)
    Sentiment(polarity=0.65, subjectivity=0.78)
    """

    def analyze_sentiment(self, text):
        """
        Analyzes the sentiment of the given text.
        
        Parameters
        ----------
        text : str
            The text to be analyzed.
        
        Returns
        -------
        Sentiment
            A named tuple containing the polarity and subjectivity of the text.
        """
        polarity, subjectivity = pattern_sentiment(text)
        return Sentiment(polarity, subjectivity)
