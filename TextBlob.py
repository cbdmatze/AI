from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer, PatternAnalyzer

class TextBlob:
    """
    A class to represent a TextBlob object.
    
    Attributes
    ----------
    text : str
        The text to be analyzed.
    analyzer : str
        The sentiment analysis method to use.
    
    Methods
    -------
    analyze_sentiment()
        Analyzes the sentiment of the text.
    
    Examples
    --------
    >>> text = "I absolutely love this phone! The battery life is amazing."
    >>> tb = TextBlob(text, analyzer='pattern')
    >>> tb.analyze_sentiment()
    'Positive'
    """
    
class TextBlob:
    def __init__(self, text, analyzer='pattern'):
        """
        Constructs a TextBlob object with the given text and sentiment analysis method.
        
        Parameters
        ----------
        text : str
            The text to be analyzed.
        analyzer : str, optional
            The sentiment analysis method to use. Default is 'pattern'.
        """
        self.text = text
        self.analyzer = analyzer
    
    def analyze_sentiment(self):
        """
        Analyzes the sentiment of the text.
        
        Returns
        -------
        str
            The sentiment of the text ('Positive', 'Negative', or 'Neutral').
        """
        if self.analyzer == 'pattern':
            blob = Blobber(analyzer=PatternAnalyzer()).blob
        elif self.analyzer == 'naive_bayes':
            blob = Blobber(analyzer=NaiveBayesAnalyzer()).blob
        else:
            raise ValueError("Invalid analyzer. Choose 'pattern' or 'naive_bayes'.")
        
        sentiment = blob(self.text).sentiment
        if sentiment.polarity > 0:
            return 'Positive'
        elif sentiment.polarity < 0:
            return 'Negative'
        else:
            return 'Neutral'