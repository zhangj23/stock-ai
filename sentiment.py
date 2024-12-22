import requests
import pandas as pd
import os
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pandas.tseries.offsets import BDay

load_dotenv()
analyzer = SentimentIntensityAnalyzer()

def combine_text(article):
   article['title'] += " " + (article["content"].split("\u2026 [", 1)[0] if article["content"] and "If you click 'Accept all'" not in article["content"] else "") + " " + (article["description"] if article["description"] else "")
   return article

def calc_score(scores):
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
      
def min_max_score(titles):
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
      data = groups.get_group(key)
      extreme_score = 0
      if data["sentiment_score"].max() > 0:
         extreme_score += data["sentiment_score"].max()

      if data["sentiment_score"].min() < 0:
         extreme_score += data["sentiment_score"].min()
      average_score = calc_score(data)
      score = average_score * 0.7 + extreme_score * 0.3
      sentiment_scores["Scores"].append(score)
   return pd.DataFrame(sentiment_scores)
      

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
      'q': ticker, # query phrase
      'sortBy': 'popularity', # articles from popular sources and publishers come first
      'pageSize': 100,  # maximum is 100 for developer version
      'apiKey': api_key, # your own API key
      'language': 'en',
      'domains': 'yahoo.com,investors.com,businessinsider.com,marketwatch.com,bloomberg.com/'
   }

   response = requests.get(url, params=parameters)

   data = pd.DataFrame(response.json())
   
   news_df = pd.concat([data['articles'].apply(pd.Series)], axis=1).apply(combine_text, axis=1)
   final_news = news_df.loc[:,['publishedAt','title']]
   final_news['publishedAt'] = pd.to_datetime(final_news['publishedAt'])
   final_news.sort_values(by='publishedAt',inplace=True)
   
   with open(f"json/{ticker}.json", "w") as outfile:
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
   ticker = "INTC"
   titles = write_info(ticker)
   titles = read_info(ticker)
   # print(titles)
   
   titles['trade_day'] = titles['publishedAt'].apply(open_trade)
   # print(titles)
   
   
   titles['sentiment_score'] = titles['title'].apply(sentiment_analyzer)
   # print(titles)
   titles.to_csv(f'csv/{ticker}_sentiment.csv', index=False)
   
   scores = min_max_score(titles)
   print(scores)
if __name__ == "__main__":
   main()