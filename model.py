from historical import *
from top_100_tickers import top_100_stocks
def main():
    model_name = "all"
    print(determine_best_seq(model_name, top_100_stocks))
    
if __name__ == "__main__":
    main()