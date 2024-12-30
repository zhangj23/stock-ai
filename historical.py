import yfinance
from technical import create_ta
from sentiment import StockSentiment
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

seq_length = 30

def etl(ticker):
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

   data = data.join(sentiment_object.scores, how='left')
   data['Scores'].fillna(0, inplace=True)

   close_data = pd.DataFrame()
   close_data['Close'] = data['Close']
   prediction_scaler = MinMaxScaler()
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
   train_size = int(cap * 0.9)
   X_train = X_scaled[:train_size]
   y_train = y_scaled[:train_size]
   X_test = X_scaled[train_size:cap]
   y_test = y_scaled[train_size:cap]
   
   return X_train, y_train, X_test, y_test, prediction_scaler, data

def create_model(X_train, y_train, ticker):
   model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length,  X_train.shape[2]), return_sequences=True),
        Dropout(0.3),
        LSTM(70, activation='relu',return_sequences=True),
        Dropout(0.4),
        LSTM(50, activation='relu'),
        Dropout(0.5),
        Dense(1)
    ])

   model.compile(optimizer='adam', loss='mse')
   model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

   model.save(f"models/{ticker}_model.keras")
   
def predict_data(X_train, X_test, prediction_scaler, ticker):
   model = tf.keras.models.load_model(f"models/{ticker}_model.keras")
   # Make predictions
   train_predictions = model.predict(X_train)
   print(X_test)
   test_predictions_scaled = model.predict(X_test)

   print(train_predictions)
   # Inverse transform the predictions
   train_predictions = prediction_scaler.inverse_transform(train_predictions)
   test_predictions = prediction_scaler.inverse_transform(test_predictions_scaled)
   
   return train_predictions, test_predictions, test_predictions_scaled

def plot_data(train_predictions, test_predictions, data, ticker):
   test_predictions = np.insert(test_predictions, 0, [train_predictions[-1]], 0)
   plt.figure(figsize=(10, 6))

   # Plot actual data
   plt.plot(data.index[seq_length:], data['Close'][seq_length:], label='Actual', color='blue')

   # Plot training predictions
   plt.plot(data.index[seq_length:seq_length+len(train_predictions)], train_predictions, label='Train Predictions',color='green')

   # Plot testing predictions
   test_pred_index = range(seq_length+len(train_predictions)-1, seq_length+len(train_predictions)+len(test_predictions)-1)
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
   good = 0
   for i in range(len(prediction)):
      x_value = x[i][-1][0]
      y_value = y[i]
      predicted_value = prediction[i][0]
      good_prediction = (x_value > y_value) == (x_value > predicted_value)
      if(good_prediction):
         good += 1
      
   print("{0:.2f}%".format(good/len(prediction)*100))


def main():
   ticker = "TSLA"
   X_train, y_train, X_test, y_test, prediction_scaler, data = etl(ticker)
   if not os.path.exists(f"models/{ticker}_model.keras"):
      create_model(X_train, y_train, ticker)
   train_predictions, test_predictions, test_predictions_scaled = predict_data(X_train, X_test, prediction_scaler, ticker)
   display_accuracy(X_test, y_test, test_predictions_scaled)
   plot_data(train_predictions, test_predictions, data, ticker)
   
if __name__ == "__main__":
   main()