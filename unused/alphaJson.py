import json
import requests

APIKEYFILE = 'alpha.txt'

def get_quarterly_data(symbol, api_key):
   url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={api_key}'
   r = requests.get(url)
   data = r.json()

   return data["quarterlyEarnings"]


def main():
   api_key = ""
   with open(APIKEYFILE) as apifile:
      api_key = apifile.readline()
   symbol = input("Ticker symbol: ")
   data = get_quarterly_data(symbol, api_key)
   json_object = json.dumps(data, indent=4)
 

   with open(f"{symbol}earnings.json", "w") as outfile:
      outfile.write(json_object)
   outfile.close()
if __name__ == "__main__":
   main()