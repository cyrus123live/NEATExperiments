import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_bollinger_bands(data, start_date=None, end_date=None):
    # If start_date and end_date are not provided, use the entire dataset
    if start_date is None:
        start_date = data.index[0]
    if end_date is None:
        end_date = data.index[-1]
    
    # Filter the data based on the date range
    plot_data = data.loc[start_date:end_date]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the closing price
    ax.plot(plot_data.index, plot_data['Close'], label='Close Price', color='blue', alpha=0.5)

    # Plot the Bollinger Bands
    ax.plot(plot_data.index, plot_data['BB_Upper'], label='Upper BB', color='red', alpha=0.7)
    ax.plot(plot_data.index, plot_data['BB_Middle'], label='Middle BB', color='green', alpha=0.7)
    ax.plot(plot_data.index, plot_data['BB_Lower'], label='Lower BB', color='red', alpha=0.7)

    # ax.plot(plot_data.index, plot_data['%D'], label='%D', color='purple', alpha=0.7)
    # ax.plot(plot_data.index, plot_data['%K'], label='%K', color='yellow', alpha=0.7)

    # Fill the area between the upper and lower bands
    ax.fill_between(plot_data.index, plot_data['BB_Upper'], plot_data['BB_Lower'], alpha=0.1)

    # Set title and labels
    ax.set_title('Stock Price with Bollinger Bands')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')

    # Format the date on the x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))  # Show date every 5 days
    plt.xticks(rotation=45)

    # Add legend
    ax.legend()

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()


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
