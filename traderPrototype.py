import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def calculate_bollinger_bands(data, window=20, num_std=2):
    # Calculate the simple moving average
    data['BB_Middle'] = data['Close'].rolling(window=window).mean()
    
    # Calculate the standard deviation
    rolling_std = data['Close'].rolling(window=window).std()
    
    # Calculate the upper and lower bands
    data['BB_Upper'] = data['BB_Middle'] + (rolling_std * num_std)
    data['BB_Lower'] = data['BB_Middle'] - (rolling_std * num_std)
    
    # Calculate the Bollinger Band width
    data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
    
    # Calculate the Bollinger Band Percentage
    data['BB_Percentage'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])

def calculate_stochastic(data, n=14, d=3):
    # Assuming 'data' is a pandas DataFrame with 'high', 'low', and 'close' columns
    low_min = data['Low'].rolling(window=n).min()
    high_max = data['High'].rolling(window=n).max()
    
    # Calculate %K
    data['%K'] = (data['Close'] - low_min) / (high_max - low_min) * 100
    
    # Calculate %D
    data['%D'] = data['%K'].rolling(window=d).mean()
    
    return data

def calculate_rsi(data, n=14):

    change = data["Close"].diff()
    change.dropna(inplace=True)

    # Create two copies of the Closing price Series
    change_up = change.copy()
    change_down = change.copy()
    change_up[change_up<0] = 0
    change_down[change_down>0] = 0

    # Verify that we did not make any mistakes
    change.equals(change_up+change_down)

    # Calculate the rolling average of average up and average down
    avg_up = change_up.rolling(14).mean()
    avg_down = change_down.rolling(14).mean().abs()

    rsi = 100 * avg_up / (avg_up + avg_down)
    data['RSI'] = rsi

def calculate_vwap(data):
    v = data['Volume'].values
    tp = (data['Low'] + data['Close'] + data['High']).div(3).values
    vwap = pd.Series(index=data.index, data=np.cumsum(tp * v) / np.cumsum(v))
    data["VWAP"] = vwap

def get_training_data():

    # Get the data of the stock
    apple = yf.Ticker('AAPL')

    # Get the historical prices for Apple stock
    historical_prices = apple.history(period='max', interval='1m')
    del historical_prices["Dividends"]
    del historical_prices["Stock Splits"]

    calculate_rsi(historical_prices)
    calculate_stochastic(historical_prices)
    calculate_vwap(historical_prices)
    calculate_bollinger_bands(historical_prices)
    historical_prices['20_Avg'] = historical_prices['Close'].rolling(window=20).mean()
    historical_prices['Price_Change'] = historical_prices['Close'].pct_change()

    historical_prices.dropna(inplace=True)

    features = ["Close", "Volume", "RSI", "20_Avg", "VWAP", "Price_Change", "BB_Upper", "BB_Lower", "BB_Width", "BB_Percentage"]
    data = pd.DataFrame(index=historical_prices.index)
    for feature in features:
        # Note: normalization introduces future information
        data[f'{feature}'] = historical_prices[feature] / historical_prices[feature].iloc[0]
        # data[f'{feature}'] = historical_prices[feature]
    data["Price_Change"] = historical_prices["Price_Change"] * 1000

    return data

held = 0
profit = 0

for index, row in get_training_data().iterrows():

    print(row.tolist())
    
    x = int(input("Your move (0: Sell All, 1: Sell 10, 2: Hold, 3: Buy 10, 4: Buy 50): "))
    if x == 0 or (x == 1 and held < 10):
        profit += held * row["Close"]
        held = 0
    elif x == 1:
        profit += 10 * row["Close"]
        held -= 10
    elif x == 3:
        profit -= 10 * row["Close"]
        held += 10
    elif x == 4:
        profit -= 50 * row["Close"]
        held += 50
    
    print(f"held: {held}, profit: {profit}")
