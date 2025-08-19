from jsonrpclib.SimpleJSONRPCServer import SimpleJSONRPCServer
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def check_sentiment(text):
    sentiment = sia.polarity_scores(text)
    compound = sentiment['compound']
    if compound >= 0.05:
        overall_sentiment = 'Positive'
    elif compound <= -0.05:
        overall_sentiment = 'Negative'
    else:
        overall_sentiment = 'Neutral'
    return overall_sentiment

def main():
    server = SimpleJSONRPCServer(('localhost',1086))
    server.register_function(check_sentiment)
    print("Server Start!")
    server.serve_forever()
    
if __name__ == "__main__":
    main()
