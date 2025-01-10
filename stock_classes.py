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

class StockETL():
   def __init__(self):
      self.ticker_list = top_100_stocks
      self.seq_length = 10
      self.train_information = {}
      self.validate_information = {}
      self.test_information = {}
      # self.train_scalers = {}
      # self.validate_scalers = {}
      self.test_scalers = {}
      
      for ticker in self.ticker_list:
         self.etl(ticker)
      
   def etl(self, ticker):
      print(ticker)
      stock_info = yfinance.Ticker(ticker)
      history = stock_info.history(period="10y", interval="1d")

      self.data = history["Close"].to_frame()
      self.data["High"] =  history["High"].to_frame()
      self.data["Low"] = history["Low"].to_frame()
      self.data["Volume"] = history["Volume"].to_frame()
      self.data = create_ta(self.data)

      self.data.drop("High", axis=1, inplace=True)
      self.data.drop("Low", axis=1, inplace=True)
      self.data.drop("Volume", axis=1, inplace=True)

      self.data.index = self.data.index.date
      
      # Create data and store in class
      cap = int(len(self.data))
      train_size = int(cap * 0.7)
      validation_cap = int(cap * 0.9)
      
      self.train_data = self.data.iloc[:train_size]
      self.validate_data = self.data.iloc[train_size:validation_cap]
      self.test_data = self.data.iloc[validation_cap:cap]
      
      train_scaler = MinMaxScaler()
      train_scaled = train_scaler.fit_transform(self.train_data)
      
      validate_scaler = MinMaxScaler()
      validate_scaled = validate_scaler.fit_transform(self.validate_data)
      
      test_scaler = MinMaxScaler()
      test_scaled = test_scaler.fit_transform(self.test_data)
      
      self.train_information[ticker] = train_scaled
      self.validate_information[ticker] = validate_scaled
      self.test_information[ticker] = test_scaled
      
      # Create scalers for reverse scaling
      self.prediction_train_scaler = MinMaxScaler()
      self.prediction_train_scaler.fit_transform(self.train_data[["Close"]])
      
      self.prediction_validate_scaler = MinMaxScaler()
      self.prediction_validate_scaler.fit_transform(self.validate_data[["Close"]])
      
      self.prediction_test_scaler = MinMaxScaler()
      self.prediction_test_scaler.fit_transform(self.test_data[["Close"]])
      
      self.test_scalers[ticker] = self.prediction_test_scaler
      
   def store_sequences(self, ticker):
      self.X_train, self.y_train = self.create_sequences(self.train_information[ticker])
      self.X_validate, self.y_validate = self.create_sequences(self.validate_information[ticker])
      self.X_test, self.y_test = self.create_sequences(self.test_information[ticker])
      
   def create_sequences(self, data):
      X, y = [], []
      for i in range(len(data) - self.seq_length):
         X.append(data[i:i+self.seq_length])
         # y.append((data[i+seq_length][0] > data[i+seq_length-1][0]).astype(int))
         y.append(data[i+self.seq_length][0])
      return np.array(X), np.array(y)
   
   def compile_etl(self):
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
      
      for ticker in self.ticker_list:
         self.store_sequences(ticker)
         if self.X_train.size:
            X_train_list.append(self.X_train)
            y_train_list.append(self.y_train)
            X_validate_list.append(self.X_validate)
            y_validate_list.append(self.y_validate)
      X_train_list = np.concatenate(X_train_list)
      y_train_list = np.concatenate(y_train_list)
      X_validate_list = np.concatenate(X_validate_list)
      y_validate_list = np.concatenate(y_validate_list)
      return X_train_list, y_train_list, X_validate_list, y_validate_list

   def change_seq_length(self, new_length):
      self.seq_length = new_length
      
      
class StockModel(StockETL):
   def __init__(self, name, batch_size):
      super().__init__()
      self.name = name
      self.batch_size = batch_size
   def create_model(self, X_train, y_train, X_validate, y_validate):
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
         Bidirectional(LSTM(64, activation='tanh', return_sequences=True), input_shape=(self.seq_length, X_train.shape[2])),
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
      model.fit(X_train, y_train, epochs=100, batch_size=self.batch_size, verbose=1, validation_data=(X_validate, y_validate), callbacks=[early_stop])

      model.save(f"models/{self.name}{self.seq_length}_model.keras")
   
   def predict_data(self, quote):
      """Predicts 2 different sets of data using a model and 
      inverses it using the Scaler passed in
      """
      model = tf.keras.models.load_model(f"models/{self.name}{self.seq_length}_model.keras")
      
      # Make predictions
      # train_predictions = model.predict(X_train)
      # validate_predictions = model.predict(X_validate)
      # X_test = self.create_sequences(self.test_information[quote])[0]
      # print(X_test)
      # print(X_test.shape)
      # print(self.seq_length)
      # print(len(X_test))
      test_predictions_scaled = model.predict(self.X_test)

      # Inverse transform the predictions
      # train_predictions = train_scaler.inverse_transform(train_predictions)
      # validate_predictions = validate_scaler.inverse_transform(validate_predictions)
      test_predictions = self.test_scalers[quote].inverse_transform(test_predictions_scaled)
      
      return test_predictions, test_predictions_scaled
   

class ModelTesting():
   def __init__(self, name, batch_size, seq_lengths):
      self.ticker_list = top_100_stocks
      self.name = name
      self.stock_model = StockModel(name, batch_size)
      self.seq_lengths = seq_lengths

   def display_accuracy(self, x, y, prediction):
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
   
   def determine_best_seq(self):
      max_percent = 0
      best_seq = 0
      dictionary = {}
      
      for i in self.seq_lengths:
         model_percent = self.model_accuracy(i)
         dictionary[i] = model_percent
         print("Accuracy for seq_length {}: {}".format(i, model_percent))
         if model_percent > max_percent:
            best_seq = i
            max_percent = model_percent
         with open('my_dict.json', 'w') as f:
            json.dump(dictionary, f)
         
            
      return "Best seq_length: {}\n The average percent: {}%".format(best_seq, max_percent)
   
   def model_accuracy(self, seq_length):
      self.stock_model.change_seq_length(seq_length)
      if not os.path.exists(f"models/{self.name}{seq_length}_model.keras"):
         X_train, y_train, X_validate, y_validate = self.stock_model.compile_etl()
         self.stock_model.create_model(X_train, y_train, X_validate, y_validate)
      
      total_amount = 0
      total_percent = 0
      for quote in self.ticker_list:
         response = self.accuracy_per_quote(quote)
         if response != None:
            total_percent += response
            total_amount += 1
      return total_percent / total_amount

   def accuracy_per_quote(self, quote):
      # response = etl(quote, seq_length)

      # if response:
      #    X_train, y_train, X_validate, y_validate, X_test, y_test, prediction_train_scaler, \
      #    prediction_validate_scaler, prediction_test_scaler = response
      # else:
      #    return None

      self.stock_model.store_sequences(quote)
      test_predictions, test_predictions_scaled = self.stock_model.predict_data(quote)
      return self.display_accuracy(self.stock_model.X_test, self.stock_model.y_test, test_predictions_scaled)