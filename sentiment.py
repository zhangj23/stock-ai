import requests
import pandas as pd
import os
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pandas.tseries.offsets import BDay

load_dotenv()
analyzer = SentimentIntensityAnalyzer()

def sentiment_analyzer(headline):
   return analyzer.polarity_scores(headline)['compound']
   
def open_trade(date):
   datetime = pd.to_datetime(date)
   today_open = pd.to_datetime(date).floor('d').replace(hour=9, minute=30) - pd.tseries.offsets.BDay(0)
   today_close = pd.to_datetime(date).floor('d').replace(hour=16, minute=0) - pd.tseries.offsets.BDay(0)
   
   next_open= pd.to_datetime(date).floor('d').replace(hour=9, minute=30) + pd.tseries.offsets.BDay()
   prev_close = pd.to_datetime(date).floor('d').replace(hour=16, minute=0) - pd.tseries.offsets.BDay()
   
   if today_open > datetime and prev_close <= datetime:
      return today_open.date()
   elif today_close < datetime and next_open > datetime:
      return next_open.date()
   else:
      return datetime.date()

def write_info(ticker):
   """Create API response and save in csv formatted for only publish time and title
   Args:
       ticker (string): Stock Ticker

   Returns:
       Panda DataFrame: DataFrame with DateTime and Headline Title
   """
   url = 'https://newsapi.org/v2/everything?'

   api_key = os.getenv("API_KEY")

   parameters = {
      'q': ticker + " stock", # query phrase
      'sortBy': 'popularity', # articles from popular sources and publishers come first
      'pageSize': 100,  # maximum is 100 for developer version
      'apiKey': api_key, # your own API key
      'language': 'en',
   }

   response = requests.get(url, params=parameters)

   print(response)
   
   data = pd.DataFrame(response.json())


   
   news_df = pd.concat([data['articles'].apply(pd.Series)], axis=1)
   print(news_df)
   final_news = news_df.loc[:,['publishedAt','title']]
   final_news['publishedAt'] = pd.to_datetime(final_news['publishedAt'])
   final_news.sort_values(by='publishedAt',inplace=True)
   
   with open(f"{ticker}.json", "w") as outfile:
      outfile.write(data.to_json())
   outfile.close()

   final_news.to_csv(f"csv/{ticker}.csv", index=False)
   return final_news
   
   
def read_info(ticker):
   """
   Read DataFrame from stored API response
   Args:
       ticker (string): Stock Ticker

   Returns:
       Panda DataFrame: DataFrame with DateTime and Headline Title
   """
   data  = pd.read_csv(f'csv/{ticker}.csv')
   print(data)
   return data


def main():
   ticker = "TSLA"
   # titles = write_info(ticker)
   titles = read_info(ticker)
   print(titles)
   
   titles['trade_day'] = titles['publishedAt'].apply(open_trade)
   print(titles)
   
   
   titles['sentiment_score'] = titles['title'].apply(sentiment_analyzer)
   print(titles)
   titles.to_csv(f'csv/{ticker}_sentiment.csv', index=False)
if __name__ == "__main__":
   main()