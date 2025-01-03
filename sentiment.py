import requests
import pandas as pd
import os
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pandas.tseries.offsets import BDay
import os
load_dotenv()

class StockSentiment:
   def __init__(self, ticker):
      self.ticker = ticker
      self.analyzer = SentimentIntensityAnalyzer()
      
      if os.path.exists(f"csv/{ticker}.csv"):
         self.data = self.read_info()
         self.merge_new_data()
      else:
         self.data = self.write_info()
      
      self.data['trade_day'] = self.data['publishedAt'].apply(self.open_trade)
   
   
      self.data['sentiment_score'] = self.data['title'].apply(self.sentiment_analyzer)
      self.data.to_csv(f'csv/{ticker}_sentiment.csv', index=False)
      
      self.scores = self.min_max_score(self.data)
      self.scores.set_index('Date', inplace=True)
   def combine_text(self, article):
      article['title'] += " " + (article["content"].split("\u2026 [", 1)[0] if article["content"] and "If you click 'Accept all'" not in article["content"] else "") + " " + (article["description"] if article["description"] else "")
      return article

   def calc_score(self, scores):
      pos = 0
      neg = 0
      length = 0
      for index, score in scores.iterrows():
         length += 1
         # print(score)
         res = (score['sentiment_score']) ** 2
         if score['sentiment_score'] > 0:
            pos += res
         else:
            neg -= res
            
      return (pos + neg)/length
         
   def min_max_score(self, titles):
      """Takes a Pandas Dataframe that includes the sentiment scores and returns a Dataframe with min_maxed scores

      Args:
         titles (Pandas DataFrame): Includes sentiment_score and trade_day at least

      Returns:
         Pandas DataFrame: With Dates paired with normalized scores
      """
      groups = titles.groupby(['trade_day'])
      dates = list(groups.groups.keys())
      sentiment_scores = {'Date': dates, 'Scores': []}
      for key in groups.groups.keys():
         data = groups.get_group((key,))
         extreme_score = 0
         if data["sentiment_score"].max() > 0:
            extreme_score += data["sentiment_score"].max()

         if data["sentiment_score"].min() < 0:
            extreme_score += data["sentiment_score"].min()
         average_score = self.calc_score(data)
         score = average_score * 0.7 + extreme_score * 0.3
         sentiment_scores["Scores"].append(score)
      return pd.DataFrame(sentiment_scores)
         
   def merge_new_data(self):
      self.data["publishedAt"] = pd.to_datetime(self.data["publishedAt"])
      url = 'https://newsapi.org/v2/everything?'

      api_key = os.getenv("API_KEY")

      parameters = {
         'q': self.ticker, # query phrase
         'sortBy': 'popularity', # articles from popular sources and publishers come first
         'pageSize': 100,  # maximum is 100 for developer version
         'apiKey': api_key, # your own API key
         'language': 'en',
         # 'domains': 'yahoo.com,investors.com,businessinsider.com,marketwatch.com,bloomberg.com/'
      }

      response = requests.get(url, params=parameters)

      data = pd.DataFrame(response.json())
      
      news_df = pd.concat([data['articles'].apply(pd.Series)], axis=1).apply(self.combine_text, axis=1)
      final_news = news_df.loc[:,['publishedAt','title']]
      final_news['publishedAt'] = pd.to_datetime(final_news['publishedAt'])
      final_news.sort_values(by='publishedAt',inplace=True)
      
      self.data = pd.concat([self.data, final_news]).drop_duplicates().reset_index(drop=True)
      print(self.data)
      self.data.sort_values(by='publishedAt',inplace=True)
      
      self.data.to_csv(f"csv/{self.ticker}.csv", index=False)
      
   def sentiment_analyzer(self, headline):
      return self.analyzer.polarity_scores(headline)['compound']
      
   def open_trade(self, date):
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

   def write_info(self):
      """Create API response and save in csv formatted for only publish time and title
      Args:
         ticker (string): Stock Ticker

      Returns:
         Panda DataFrame: DataFrame with DateTime and Headline Title
      """
      url = 'https://newsapi.org/v2/everything?'

      api_key = os.getenv("API_KEY")

      parameters = {
         'q': self.ticker, # query phrase
         'sortBy': 'popularity', # articles from popular sources and publishers come first
         'pageSize': 100,  # maximum is 100 for developer version
         'apiKey': api_key, # your own API key
         'language': 'en',
         # 'domains': 'yahoo.com,investors.com,businessinsider.com,marketwatch.com,bloomberg.com/'
      }

      response = requests.get(url, params=parameters)

      data = pd.DataFrame(response.json())
      
      news_df = pd.concat([data['articles'].apply(pd.Series)], axis=1).apply(self.combine_text, axis=1)
      final_news = news_df.loc[:,['publishedAt','title']]
      final_news['publishedAt'] = pd.to_datetime(final_news['publishedAt'])
      final_news.sort_values(by='publishedAt',inplace=True)
      
      with open(f"json/{self.ticker}.json", "w") as outfile:
         outfile.write(data.to_json())
      outfile.close()

      final_news.to_csv(f"csv/{self.ticker}.csv", index=False)
      return final_news
      
      
   def read_info(self):
      """
      Read DataFrame from stored API response
      Args:
         ticker (string): Stock Ticker

      Returns:
         Panda DataFrame: DataFrame with DateTime and Headline Title
      """
      data  = pd.read_csv(f'csv/{self.ticker}.csv')
      return data


def main():
   ticker = "INTC"
   sentiment_class = StockSentiment(ticker)
   # print(sentiment_class.scores)
   
if __name__ == "__main__":
   main()