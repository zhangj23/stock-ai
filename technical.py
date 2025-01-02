import ta

def create_ta(data):
   # print(data)
   data['SMA50'] = ta.trend.sma_indicator(data['Close'], window=50)
   data['EMA20'] = ta.trend.sma_indicator(data['Close'], window=20)
   
   data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
   
   data['MACD'] = ta.trend.macd(data['Close'])
   data['MACD_Signal'] = ta.trend.macd_signal(data['Close'])
   data['MACD_Hist'] = ta.trend.macd_diff(data['Close'])
   
   data['BB_High'] = ta.volatility.bollinger_hband(data['Close'], window=20)
   data['BB_Low'] = ta.volatility.bollinger_lband(data['Close'], window=20)
   
   data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=14)

   data['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
   
   data.dropna(inplace=True)
   # print(data)
   
   return data

