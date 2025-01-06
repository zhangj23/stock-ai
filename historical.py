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

seq_length = 20

def etl(ticker):
   print(ticker)
   stock_info = yfinance.Ticker(ticker)
   history = stock_info.history(period="2y", interval="1d")

   data = history["Close"].to_frame()
   data["High"] =  history["High"].to_frame()
   data["Low"] = history["Low"].to_frame()
   data["Volume"] = history["Volume"].to_frame()
   data = create_ta(data)

   data.drop("High", axis=1, inplace=True)
   data.drop("Low", axis=1, inplace=True)

   data.index = data.index.date

   sentiment_object = StockSentiment(ticker)

   if sentiment_object.scores.empty:
      print("ok")
      return None
   data = data.join(sentiment_object.scores, how='left')
   
   # Replace rows with no sentiment with 0
   # data['Scores'].fillna(0, inplace=True)
   
   # Drop rows with empty sentiment
   # data.dropna(inplace=True) 
   
   sentiment_average = data['Scores'].mean()
   first_sentiment_not_found = True
   prev_score = 0
   for index in data.index:
      if pd.isna(data.at[index, 'Scores']) and first_sentiment_not_found:
         data.at[index, 'Scores'] = sentiment_average
      elif pd.isna(data.at[index, 'Scores']):
         data.at[index, 'Scores'] = prev_score
      else:
         prev_score = data.at[index, 'Scores']
         first_sentiment_not_found = False
   close_data = pd.DataFrame()
   close_data['Close'] = data['Close']
   prediction_scaler = MinMaxScaler()
   
   if(ticker == "ABT"):
      print(close_data)
      
   prediction_scaled_data = prediction_scaler.fit_transform(close_data)
   
   scaler = MinMaxScaler()
   scaled_data = scaler.fit_transform(data)


   # cap = int(len(scaled_data) * 1)
   # train_size = int(cap * 0.9)
   # train_data = scaled_data[:train_size]
   # test_data = scaled_data[train_size:cap]

   # X_train, y_train = create_sequences(train_data, seq_length)
   # X_test, y_test = create_sequences(test_data, seq_length)


   X_scaled, y_scaled = create_sequences(scaled_data, seq_length)
   
   cap = int(len(X_scaled) * 1)
   train_size = int(cap * 0.7)
   validation_cap = int(cap * 0.9)
   X_train = X_scaled[:train_size]
   y_train = y_scaled[:train_size]
   X_validate = X_scaled[train_size:validation_cap]
   y_validate = y_scaled[train_size:validation_cap]
   X_test = X_scaled[validation_cap:cap]
   y_test = y_scaled[validation_cap:cap]
   
   return X_train, y_train, X_test, y_test, prediction_scaler, data, X_scaled, y_scaled, X_validate, y_validate

def create_model(X_train, y_train, X_validation, y_validation, ticker):
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


   model.compile(optimizer='adam', loss='mse')
   early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
   model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1, validation_data=(X_validation, y_validation), callbacks=[early_stop])

   model.save(f"models/{ticker}{seq_length}_model.keras")
   
def predict_data(X_train, X_test, X_validate, prediction_scaler, ticker):
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
   print(X_test)
   test_predictions_scaled = model.predict(X_test)

   print(train_predictions)
   # Inverse transform the predictions
   train_predictions = prediction_scaler.inverse_transform(train_predictions)
   validate_predictions = prediction_scaler.inverse_transform(validate_predictions)
   test_predictions = prediction_scaler.inverse_transform(test_predictions_scaled)
   
   return train_predictions, test_predictions, test_predictions_scaled, validate_predictions

def plot_data(train_predictions, test_predictions, validate_prediction, data, ticker):
   test_predictions = np.insert(test_predictions, 0, [validate_prediction[-1]], 0)
   plt.figure(figsize=(10, 6))

   # Plot actual data
   plt.plot(data.index[seq_length:], data['Close'][seq_length:], label='Actual', color='blue')

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
      
   print("{0:.2f}%".format(good/len(prediction)*100))

def compile_etl(ticker_list):
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
      response = etl(ticker)
      if response:
         X_train, y_train, X_test, y_test, prediction_scaler, data, X_scaled, y_scaled, X_validate, y_validate = response
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
def main():
   ticker = "all"
   test_ticker = "NVDA"
   # ticker_list = ["TSLA", "NVDA", "AAPL", "QQQ", "SPY", "AMZN", "VOO", "GOOGL", "MSFT", "META", "MS", "GS", "VZ", "NFLX", "COST", "PG", "KO", "JNJ"]
   ticker_list = top_100_stocks
   # X_train, y_train, X_test, y_test, prediction_scaler, data = etl(ticker)
   
   X_train_sample, _, X_test, y_test, prediction_scaler, data, X_scaled, y_scaled, X_validate_test_ticker, _ = etl(test_ticker)
   if not os.path.exists(f"models/{ticker}{seq_length}_model.keras"):
      X_train, y_train, X_validate, y_validate = compile_etl(ticker_list)
      create_model(X_train, y_train, X_validate, y_validate, ticker)
      
   # Use this for a stock not in the ticker list
   # train_predictions, test_predictions, test_predictions_scaled = predict_data(X_train_sample, X_scaled, prediction_scaler, ticker)
   # display_accuracy(X_scaled, y_scaled, test_predictions_scaled)
   
   # Use this for a stock in the ticker list
   train_predictions, test_predictions, test_predictions_scaled, validate_predictions = predict_data(X_train_sample, X_test, X_validate_test_ticker, prediction_scaler, ticker)
   display_accuracy(X_test, y_test, test_predictions_scaled)
   
   plot_data(train_predictions, test_predictions, validate_predictions, data, test_ticker)
   
if __name__ == "__main__":
   main()