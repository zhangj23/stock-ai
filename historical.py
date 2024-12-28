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

def create_model(ticker):
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
   print(prediction_scaled_data)
   
   scaler = MinMaxScaler()
   scaled_data = scaler.fit_transform(data)

   print(scaled_data)

   cap = int(len(scaled_data) * 1)
   train_size = int(cap * 0.95)
   train_data = scaled_data[:train_size]
   test_data = scaled_data[train_size+1:cap+1]

   seq_length = 10  # Number of time steps to look back
   X_train, y_train = create_sequences(train_data, seq_length)
   X_test, y_test = create_sequences(test_data, seq_length)

   # print(f"X_test shape: {X_test.shape}")
   # if X_test.shape[0] == 0:
   #    raise ValueError("X_test is empty. Check your data preparation steps.")
   
   model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length,  X_train.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(70, activation='relu', return_sequences = True),
        Dropout(0.3),
        LSTM(100, activation='relu'),
        Dropout(0.4),
        Dense(1)
    ])

   model.compile(optimizer='adam', loss='mse')
   model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

   # Make predictions
   train_predictions = model.predict(X_train)
   test_predictions = model.predict(X_test)

   # Inverse transform the predictions
   train_predictions = prediction_scaler.inverse_transform(train_predictions)
   test_predictions = prediction_scaler.inverse_transform(test_predictions)
   
   plt.figure(figsize=(10, 6))

   # Plot actual data
   plt.plot(data.index[seq_length:], data['Close'][seq_length:], label='Actual', color='blue')

   # Plot training predictions
   plt.plot(data.index[seq_length:seq_length+len(train_predictions)], train_predictions, label='Train Predictions',color='green')

   # Plot testing predictions
   test_pred_index = range(seq_length+len(train_predictions), seq_length+len(train_predictions)+len(test_predictions))
   plt.plot(data.index[test_pred_index], test_predictions, label='Test Predictions',color='orange')

   plt.title('Money')
   plt.xlabel('Year')
   plt.ylabel('NVDA stock')
   plt.show()
   
def create_sequences(data, seq_length):
   X, y = [], []
   for i in range(len(data) - seq_length):
      X.append(data[i:i+seq_length])
      y.append(data[i+seq_length][0])
   return np.array(X), np.array(y)

def main():
   ticker = "AMD"
   create_model(ticker)
   
if __name__ == "__main__":
   main()