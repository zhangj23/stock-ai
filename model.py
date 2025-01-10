# from historical import *
# from top_100_tickers import top_100_stocks
# def main():
#     model_name = "drop"
#     print(determine_best_seq(model_name, top_100_stocks))

from stock_classes import *


def main():
    name = "drop"
    
    testing_object = ModelTesting(name, 64, [5, 10, 20, 30])
    print(testing_object.determine_best_seq())

if __name__ == "__main__":
    main()