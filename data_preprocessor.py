from sklearn.feature_extraction.text import TfidfVectorizer


def parse():
    """Perform a variety of tasks to parse the raw text into usable format
    1. Remove first and last sentences from each page of each book, and ignore first 10% and last 6%.
    2. Remove punctuation and lower case every word"""
    pass

def create_vocabulary():
    """Process the words with TFIDF across the time domains and get top tfidf terms to make our vocabulary"""
    pass

def get_raw_counts():
    """For each term in the top TFIDF terms, get the raw counts"""
    pass


parse()