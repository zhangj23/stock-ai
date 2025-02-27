import yfinance
from technical import create_ta
from sentiment import StockSentiment
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
from tensorflow.keras.callbacks import EarlyStopping
from top_100_tickers import top_100_stocks
import json
from tensorflow.keras.optimizers import Adam


def etl(ticker, seq_length):
   print(ticker)
   stock_info = yfinance.Ticker(ticker)
   history = stock_info.history(period="10y", interval="1d")

   data = history["Close"].to_frame()
   data["High"] =  history["High"].to_frame()
   data["Low"] = history["Low"].to_frame()
   data["Volume"] = history["Volume"].to_frame()
   data = create_ta(data)

   data.drop("High", axis=1, inplace=True)
   data.drop("Low", axis=1, inplace=True)
   data.drop("Volume", axis=1, inplace=True)

   data.index = data.index.date

   # sentiment_object = StockSentiment(ticker)

   # if sentiment_object.scores.empty:
   #    print("ok")
   #    return None
   # data = data.join(sentiment_object.scores, how='left')
   
   # data.ffill(inplace=True)
   # data.bfill(inplace=True)
   # Replace rows with no sentiment with 0
   # data['Scores'] = data['Scores'].fillna(0)
   
   # Drop rows with empty sentiment
   # data.dropna(inplace=True) 
   
   # Uses previous sentiment, if no previous, then use the mean
   # sentiment_average = data['Scores'].mean()
   # first_sentiment_not_found = True
   # prev_score = 0
   # for index in data.index:
   #    if pd.isna(data.at[index, 'Scores']) and first_sentiment_not_found:
   #       data.at[index, 'Scores'] = sentiment_average
   #    elif pd.isna(data.at[index, 'Scores']):
   #       data.at[index, 'Scores'] = prev_score
   #    else:
   #       prev_score = data.at[index, 'Scores']
   #       first_sentiment_not_found = False
   



   # Scale all columns relative to 'Close' range
   # scaled_data = data.copy()
   # for col in data.columns:
   #    if col != 'Close':
   #       scaled_data[col] = (data[col] - data['Close'].min()) / (data['Close'].max() - data['Close'].min())
   # scaled_data['Close'] = close_scaled 

   # Extract 'Close' column for prediction scaling
   prediction_scaler = MinMaxScaler()
   prediction_scaled_data = prediction_scaler.fit_transform(data[['Close']])

   # correlation_matrix = data.corr()
   # print("\nCorrelation Matrix:\n", correlation_matrix)
   # print("\nCorrelation with 'Close':\n", correlation_matrix['Close'])
   
   
   
   # cap = int(len(scaled_data) * 1)
   # train_size = int(cap * 0.9)
   # train_data = scaled_data[:train_size]
   # test_data = scaled_data[train_size:cap]

   # X_train, y_train = create_sequences(train_data, seq_length)
   # X_test, y_test = create_sequences(test_data, seq_length)


   
   
   # cap = int(len(X_scaled) * 1)
   # train_size = int(cap * 0.7)
   # validation_cap = int(cap * 0.9)
   # X_train = X_scaled[:train_size]
   # y_train = y_scaled[:train_size]
   # X_validate = X_scaled[train_size:validation_cap]
   # y_validate = y_scaled[train_size:validation_cap]
   # X_test = X_scaled[validation_cap:cap]
   # y_test = y_scaled[validation_cap:cap]
   
   cap = int(len(data) * 1)
   train_size = int(cap * 0.7)
   validation_cap = int(cap * 0.9)
   train_data = data.iloc[:train_size]
   validate_data = data.iloc[train_size:validation_cap]
   test_data = data.iloc[validation_cap:cap]
   
   train_scaler = MinMaxScaler()
   train_scaled = train_scaler.fit_transform(train_data)
   
   validate_scaler = MinMaxScaler()
   validate_scaled = validate_scaler.fit_transform(validate_data)
   
   test_scaler = MinMaxScaler()
   test_scaled = test_scaler.fit_transform(test_data)
   
   prediction_train_scaler = MinMaxScaler()
   prediction_train_scaler.fit_transform(train_data[["Close"]])
   
   prediction_validate_scaler = MinMaxScaler()
   prediction_validate_scaler.fit_transform(validate_data[["Close"]])
   
   prediction_test_scaler = MinMaxScaler()
   prediction_test_scaler.fit_transform(test_data[["Close"]])
   
   X_train, y_train = create_sequences(train_scaled, seq_length)
   X_validate, y_validate = create_sequences(validate_scaled, seq_length)
   X_test, y_test = create_sequences(test_scaled, seq_length)   
   
   
   return X_train, y_train, X_validate, y_validate, X_test, y_test, prediction_train_scaler, \
      prediction_validate_scaler, prediction_test_scaler

def create_model(X_train, y_train, X_validation, y_validation, ticker, seq_length):
   # model = Sequential([
   #      LSTM(50, activation='relu', input_shape=(seq_length,  X_train.shape[2]), return_sequences=True),
   #      Dropout(0.3),
   #      LSTM(70, activation='relu',return_sequences=True),
   #      Dropout(0.4),
   #      LSTM(50, activation='relu'),
   #      Dropout(0.5),
   #      Dense(1)
   #  ])

   model = Sequential([
      Bidirectional(LSTM(64, activation='tanh', return_sequences=True), input_shape=(seq_length, X_train.shape[2])),
      Dropout(0.3),
      Bidirectional(LSTM(128, activation='tanh', return_sequences=False)),
      Dropout(0.5),
      Dense(1)
   ])


   # model = Sequential()

   # model.add(LSTM(units=50,return_sequences=True,input_shape=(seq_length, X_train.shape[2])))
   # model.add(Dropout(0.2))
   # model.add(LSTM(units=50,return_sequences=True))
   # model.add(Dropout(0.2))
   # model.add(LSTM(units=50,return_sequences=True))
   # model.add(Dropout(0.2))
   # model.add(LSTM(units=50))
   # model.add(Dropout(0.2))
   # model.add(Dense(units=1))

   optimizer = Adam(learning_rate=0.001)
   model.compile(optimizer=optimizer, loss='mse')
   
   early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
   model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1, validation_data=(X_validation, y_validation), callbacks=[early_stop])

   model.save(f"models/{ticker}{seq_length}_model.keras")
   
def predict_data(X_train, X_test, X_validate, ticker, seq_length, train_scaler, test_scaler, validate_scaler):
   """Predicts 2 different sets of data using a model and 
   inverses it using the Scaler passed in

   Args:
       X_train (np array): Set 1 (training data)
       X_test (np array): Set 2 (test data)
       prediction_scaler (MinMaxScaler()): sk learn MinMax
       ticker (string): Ticker of the model you want to use

   Returns:
       tuple: set 1 predictions, set 2 predictions, set 2 scaled predictions
   """
   model = tf.keras.models.load_model(f"models/{ticker}{seq_length}_model.keras")
   # Make predictions
   train_predictions = model.predict(X_train)
   validate_predictions = model.predict(X_validate)
   test_predictions_scaled = model.predict(X_test)

   # Inverse transform the predictions
   train_predictions = train_scaler.inverse_transform(train_predictions)
   validate_predictions = validate_scaler.inverse_transform(validate_predictions)
   test_predictions = test_scaler.inverse_transform(test_predictions_scaled)
   
   return train_predictions, test_predictions, test_predictions_scaled, validate_predictions

def plot_data(train_predictions, test_predictions, validate_prediction, data, ticker, seq_length):
   test_predictions = np.insert(test_predictions, 0, [validate_prediction[-1]], 0)
   plt.figure(figsize=(10, 6))

   # Plot actual data
   plt.plot(data.index[seq_length:], data['Close'][seq_length:], label='Actual', color='blue')
   
   # plt.plot(data.index[seq_length:], data['RSI'][seq_length:], label='RSI', color='yellow')
   
   # plt.plot(data.index[seq_length:], data['MACD_Hist'][seq_length:], label='HIST', color='purple')
   
   # plt.plot(data.index[seq_length:], data['Volume'][seq_length:], label='Volume', color='red')

   # Plot training predictions
   plt.plot(data.index[seq_length:seq_length+len(train_predictions)], train_predictions, label='Train Predictions',color='green')

   test_pred_index = range(seq_length+len(train_predictions) -1, seq_length+len(train_predictions)+len(validate_prediction)-1)
   plt.plot(data.index[test_pred_index], validate_prediction, label='Validate Predictions',color='black')
   
   # Plot testing predictions
   test_pred_index = range(seq_length+len(train_predictions)+ len(validate_prediction)-1, seq_length+len(train_predictions)+len(test_predictions)+ len(validate_prediction)-1)
   plt.plot(data.index[test_pred_index], test_predictions, label='Test Predictions',color='orange')


   plt.title('Money')
   plt.xlabel('Year')
   plt.ylabel(f'{ticker} stock')
   plt.show()
   
def create_sequences(data, seq_length):
   X, y = [], []
   for i in range(len(data) - seq_length):
      X.append(data[i:i+seq_length])
      # y.append((data[i+seq_length][0] > data[i+seq_length-1][0]).astype(int))
      y.append(data[i+seq_length][0])
   return np.array(X), np.array(y)

def display_accuracy(x, y, prediction):
   """Returns a percentage based on how accurate the predictor was on the stock going up or down

   Args:
       x (np array): Original X with all sequences
       y (np array): Original y with correct returns
       prediction (np array): Predicted y by model
   """
   good = 0
   for i in range(len(prediction)):
      x_value = x[i][-1][0]
      y_value = y[i]
      predicted_value = prediction[i][0]
      good_prediction = (x_value > y_value) == (x_value > predicted_value)
      if(good_prediction):
         good += 1
      
   percent = good/len(prediction)*100
   print("{0:.2f}%".format(percent))

   return percent
   
def compile_etl(ticker_list, seq_length):
   """Uses the etl function to combine a bunch of X and Y 
   training and testing data to make a model based on more stocks

   Args:
       ticker_list (array): list of tickers to be used

   Returns:
       tuple: list of X_train and y_train lists that are already MinMaxed
   """
   X_train_list = []
   y_train_list = []
   X_validate_list = []
   y_validate_list = []
   
   for ticker in ticker_list:
      response = etl(ticker, seq_length)
      if response:
         X_train, y_train, X_validate, y_validate, X_test, y_test, prediction_train_scaler, \
      prediction_validate_scaler, prediction_test_scaler = response
      else:
         continue
      if X_train.size:
         X_train_list.append(X_train)
         y_train_list.append(y_train)
         X_validate_list.append(X_validate)
         y_validate_list.append(y_validate)
      # X_test_list = np.concatenate((X_test_list, X_test))
      # y_test_list = np.concatenate((y_test_list, y_test))
   X_train_list = np.concatenate(X_train_list)
   y_train_list = np.concatenate(y_train_list)
   X_validate_list = np.concatenate(X_validate_list)
   y_validate_list = np.concatenate(y_validate_list)
   return X_train_list, y_train_list, X_validate_list, y_validate_list

def model_accuracy(model_name, seq_length, ticker_list):
   if not os.path.exists(f"models/{model_name}{seq_length}_model.keras"):
      X_train, y_train, X_validate, y_validate = compile_etl(ticker_list, seq_length)
      create_model(X_train, y_train, X_validate, y_validate, model_name, seq_length)
      
   total_amount = 0
   total_percent = 0
   for quote in ticker_list:
      response = accuracy_per_quote(quote, model_name, seq_length)
      if response != None:
         total_percent += response
         total_amount += 1
   return total_percent / total_amount

def accuracy_per_quote(quote, model_name, seq_length):
   response = etl(quote, seq_length)

   if response:
      X_train, y_train, X_validate, y_validate, X_test, y_test, prediction_train_scaler, \
      prediction_validate_scaler, prediction_test_scaler = response
   else:
      return None
   train_predictions, test_predictions, test_predictions_scaled, validate_predictions = predict_data(X_train, X_test, X_validate, model_name, seq_length, prediction_train_scaler, \
      prediction_validate_scaler, prediction_test_scaler)
   return display_accuracy(X_test, y_test, test_predictions_scaled)

def determine_best_seq(model_name, ticker_list):
   max_percent = 0
   best_seq = 0
   dictionary = {}
   seq_lengths = [5, 10, 20, 30, 60]
   for i in seq_lengths:
      model_percent = model_accuracy(model_name, i, ticker_list)
      dictionary[i] = model_percent
      print("Accuracy for seq_length {}: {}".format(i, model_percent))
      if model_percent > max_percent:
         best_seq = i
         max_percent = model_percent
      with open('my_dict.json', 'w') as f:
         json.dump(dictionary, f)
      
         
   return "Best seq_length: {}\n The average percent: {}%".format(best_seq, max_percent)

def main():
   model_name = "drop"
   test_ticker = "TSLA"
   seq_length = 30
   
   # ticker_list = ["TSLA", "NVDA", "AAPL", "QQQ", "SPY", "AMZN", "VOO", "GOOGL", "MSFT", "META", "MS", "GS", "VZ", "NFLX", "COST", "PG", "KO", "JNJ"]
   ticker_list = top_100_stocks
   # X_train, y_train, X_test, y_test, prediction_scaler, data = etl(ticker)
   
   X_train_sample, _, X_test, y_test, prediction_scaler, data, X_scaled, y_scaled, X_validate_test_ticker, _ = etl(test_ticker, seq_length)
   if not os.path.exists(f"models/{model_name}{seq_length}_model.keras"):
      X_train, y_train, X_validate, y_validate = compile_etl(ticker_list, seq_length)
      create_model(X_train, y_train, X_validate, y_validate, model_name, seq_length)
      
   # Use this for a stock not in the ticker list
   # train_predictions, test_predictions, test_predictions_scaled = predict_data(X_train_sample, X_scaled, prediction_scaler, ticker)
   # display_accuracy(X_scaled, y_scaled, test_predictions_scaled)

   # Use this for a stock in the ticker list
   train_predictions, test_predictions, test_predictions_scaled, validate_predictions = predict_data(X_train_sample, X_test, X_validate_test_ticker, prediction_scaler, model_name, seq_length)
   display_accuracy(X_test, y_test, test_predictions_scaled)
   
   plot_data(train_predictions, test_predictions, validate_predictions, data, test_ticker, seq_length)
   
   
if __name__ == "__main__":
   main()