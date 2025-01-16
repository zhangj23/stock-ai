# from historical import *
# from top_100_tickers import top_100_stocks
# def main():
#     model_name = "drop"
#     print(determine_best_seq(model_name, top_100_stocks))

from stock_classes import *

def main():
    choice = input("Which operation do you want to do? (a for accuracy, p for plot): ")
    if choice == "a":
        name = "drop"
        
        testing_object = ModelTesting(name, 64, [5, 10, 20, 30])
        print(testing_object.determine_best_seq())
    elif choice == "p":
        name = "drop"
        quote = "AAPL"
        plot_object = PlotPredictions(name, 64, quote, 30)
        plot_object.run()
    else:
        print("Invalid input")
        
if __name__ == "__main__":
    main()