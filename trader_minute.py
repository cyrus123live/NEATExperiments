import random
import math
from collections import deque
import requests
import sqlite3

import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from alive_progress import alive_bar
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

TICKER = "BTC-USD"
NUM_INPUT_NODES = 11
DB_NAME = 'trader_database.db'

conn = sqlite3.connect(DB_NAME)
# conn.execute('''DROP TABLE days''')
conn.execute('''
    CREATE TABLE IF NOT EXISTS days (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        close REAL,
        cash REAL,
        decision INTEGER,
        held INTEGER,
        value REAL
    )
''')

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

    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': 'BTCUSDT',
        'interval': '1m',
        'limit': '100'  # Number of data points
    }

    response = requests.get(url, params=params)
    data = response.json()

    historical_prices = {
        "Time": [],
        "Close": [],
        "Volume": [],
        "Low": [],
        "High": []
    }

    print(f"Candle timestamp: {datetime.fromtimestamp(data[-2][0] / 1000.0)} to {datetime.fromtimestamp(data[-2][6] / 1000.0)}")
    print(f"Closing price: {data[-2][4]}")

    for candle in data:
        historical_prices["Time"].append(datetime.fromtimestamp(candle[0] / 1000.0))
        historical_prices["Close"].append(float(candle[4]))
        historical_prices["Volume"].append(float(candle[5]))
        historical_prices["Low"].append(float(candle[3]))
        historical_prices["High"].append(float(candle[2]))

    historical_prices = pd.DataFrame(historical_prices)

    calculate_rsi(historical_prices)
    calculate_stochastic(historical_prices)
    calculate_vwap(historical_prices)
    calculate_bollinger_bands(historical_prices)
    historical_prices['20_Avg'] = historical_prices['Close'].rolling(window=20).mean()
    historical_prices['Price_Change'] = historical_prices['Close'].pct_change()

    features = ["Close", "Volume", "RSI", "20_Avg", "VWAP", "Price_Change", "BB_Upper", "BB_Lower", "BB_Width", "BB_Percentage"]
    data = pd.DataFrame(index=historical_prices.index)

    # Normalize each feature using a rolling window
    for feature in features:
        rolling_mean = historical_prices[feature].rolling(window=20).mean()
        rolling_std = historical_prices[feature].rolling(window=20).std()

        # Normalize the feature (subtract rolling mean, divide by rolling std dev)
        data[f'{feature}'] = (historical_prices[feature] - rolling_mean) / rolling_std
    data["Close_Not_Normalized"] = historical_prices["Close"]

    # data.dropna(inplace=True)

    return data

def produce_move(c, inputs):

    q = deque()

    for i, input_node in enumerate(c["nodes"][1:NUM_INPUT_NODES+1]):
        input_node["value"] = inputs[i]
    
    for input_node in c["nodes"][:NUM_INPUT_NODES+1]:
        q.append(input_node)

    while q:
        node = q.popleft()
        edgesOut = [e for e in c["edges"] if e["source"] == node["id"]]
        for e in edgesOut:
            destination_node = [n for n in c["nodes"] if n["id"] == e["dest"]][0]
            destination_node["value"] += node["value"] * e["weight"]
            q.append(destination_node)

    result = [n["value"] for n in c["nodes"] if n["layer"] == "output"]

    # Reset system
    for node in c["nodes"][1:]:
        node["value"] = 0

    return result

date = datetime.now()
trader_string = ""
while True:
    try:
        with open(f'models/{date.strftime("%Y-%m-%d")}.txt', 'r') as file:
            trader_string = file.read()
        break
    except:
        date -= timedelta(days=1)
        continue

node_strings = [s for s in trader_string.split("edges")[0].split("\n")[2:] if len(s) > 0]
edge_strings = [s for s in trader_string.split("edges")[1].split("\n")[2:] if len(s) > 0]
nodes = []
edges = []
for n in node_strings:
    nodes.append({
        "id": int(n.split(":")[1].split(",")[0]),
        "value": 0,
        "layer": n.split(":")[3].split("'")[1]
    })

for e in edge_strings:
    edge = {
        "source": int(e.split(":")[1].split(",")[0]),
        "dest": int(e.split(":")[2].split(",")[0]),
        "weight": float(e.split(":")[3].split(",")[0]),
        "enabled": e.split(":")[4].split(",")[0].replace(" ", "")
    }
    if edge["enabled"] == 'False':
        edge["enabled"] = False
    else:
        edge["enabled"] = True
    edges.append(edge)

try:
    previous_day = conn.execute('SELECT * FROM days').fetchall()[-1]
except:
    previous_day = [0, 0, 0, 1000000, 0, 0]

t = {
    "held": previous_day[5],
    "cash": previous_day[3],
    "phenotype": {
        "nodes": nodes,
        "edges": edges
    }
}

row = get_training_data().iloc[-2] # -1 is unfinished current minute (low volume unless at end of current minute), I think start at start of minute using last minutes data to remove possibility of starting at end but getting new minutes very starting data

inputs = row.tolist() + [float((t["held"] * row["Close_Not_Normalized"]) / (t["held"] * row["Close_Not_Normalized"] + t["cash"]))] # Add amount held as proportion of portfolio value to inputs
inputs.remove(row["Close_Not_Normalized"])
decision = produce_move(t["phenotype"], inputs)
decision_index = decision.index(max(decision))
if decision_index == 0 or (decision_index == 1 and t["held"] < 10):
    t["cash"] += t["held"] * row["Close_Not_Normalized"]
    t["held"] = 0
    print("Sell All")
elif decision_index == 1:
    t["cash"] += 1 * row["Close_Not_Normalized"]
    t["held"] -= 1
    print("Sell")
elif decision_index == 3 and t["cash"] >= 1 * row["Close_Not_Normalized"]:
    t["cash"] -= 1 * row["Close_Not_Normalized"]
    t["held"] += 1
    print("Buy")
elif decision_index == 4 and t["cash"] >= 10 * row["Close_Not_Normalized"]:
    t["cash"] -= 10 * row["Close_Not_Normalized"]
    t["held"] += 10
    print("Buy a Lot")

conn.execute('''
    INSERT INTO days (date, close, cash, decision, held, value) VALUES (?, ?, ?, ?, ?, ?)
''', (date, row["Close_Not_Normalized"], t["cash"], decision_index, t["held"], t["held"] * row["Close_Not_Normalized"] + t["cash"]))

conn.commit()
conn.close()
