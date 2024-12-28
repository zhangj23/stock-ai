
# print(yfinance.get_analysts_info())

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pandas as pd

def create_model():
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length, 1), dropout=0.01),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

    # Make predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # Inverse transform the predictions
    train_predictions = scaler.inverse_transform(train_predictions)
    test_predictions = scaler.inverse_transform(test_predictions)

# Plot predictions
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

forecast_period = 30


# Use the last sequence from the test data to make predictions
last_sequence = X_test[-1]
forecast = [last_sequence[-1][-1]]
for _ in range(forecast_period):
    # Reshape the sequence to match the input shape of the model
    current_sequence = last_sequence.reshape(1, seq_length, 1)
    # Predict the next value
    print(last_sequence)
    n = model.predict(current_sequence)
    print(n)
    next_prediction = n[0][0]
    # Append the prediction to the forecast list
    forecast.append(next_prediction)
    # Update the last sequence by removing the first element and appending the predicted value
    last_sequence = np.append(last_sequence[1:], next_prediction)

# Inverse transform the forecasted values
forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

# Plot the forecasted values
plt.figure(figsize=(10, 6))
plt.plot(data.index[-len(test_data):], scaler.inverse_transform(test_data), label='Actual')
plt.plot(pd.date_range(start=data.index[-1], periods=forecast_period+1), forecast, label='Forecast')
plt.title('Air Passengers Time Series Forecasting (30-day Forecast)')
plt.xlabel('Year')
plt.ylabel('Number of Passengers')
plt.legend()
plt.show()