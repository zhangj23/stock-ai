import requests
import pandas as pd
import os
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()

def open_trade(date):
   
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
      'q': ticker, # query phrase
      'sortBy': 'popularity', # articles from popular sources and publishers come first
      'pageSize': 100,  # maximum is 100 for developer version
      'apiKey': api_key, # your own API key
      'language': 'en'
   }

   response = requests.get(url, params=parameters)

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
   
   analyzer = SentimentIntensityAnalyzer()
   
   titles['']
if __name__ == "__main__":
   main()