import random
import math
from collections import deque

import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from jaal import Jaal
from jaal.datasets import load_got

from alive_progress import alive_bar

TICKER_SYMBOL = "BTC-USD"
START_DATE = "2024-08-17"
END_DATE = "2024-08-17"

NUM_INPUT_NODES = 11
STARTING_CASH = 1000000

trader_string = '''
nodes:
{'id': 0, 'value': 1, 'layer': 'input'}
{'id': 1, 'value': 0, 'layer': 'input'}
{'id': 2, 'value': 0, 'layer': 'input'}
{'id': 3, 'value': 0, 'layer': 'input'}
{'id': 4, 'value': 0, 'layer': 'input'}
{'id': 5, 'value': 0, 'layer': 'input'}
{'id': 6, 'value': 0, 'layer': 'input'}
{'id': 7, 'value': 0, 'layer': 'input'}
{'id': 8, 'value': 0, 'layer': 'input'}
{'id': 9, 'value': 0, 'layer': 'input'}
{'id': 10, 'value': 0, 'layer': 'input'}
{'id': 12, 'value': 0, 'layer': 'output'}
{'id': 13, 'value': 0, 'layer': 'output'}
{'id': 14, 'value': 0, 'layer': 'output'}
{'id': 15, 'value': 0, 'layer': 'output'}
{'id': 16, 'value': 0, 'layer': 'output'}

edges:
{'source': 0, 'dest': 12, 'weight': 1.4069946309184682, 'enabled': True, 'innovationNum': 0}
{'source': 1, 'dest': 12, 'weight': -0.7732803824987462, 'enabled': True, 'innovationNum': 1}
{'source': 2, 'dest': 12, 'weight': 1.4552255067099962, 'enabled': True, 'innovationNum': 2}
{'source': 3, 'dest': 12, 'weight': 0.7591798871321482, 'enabled': True, 'innovationNum': 3}
{'source': 4, 'dest': 12, 'weight': 1.0603605090646213, 'enabled': True, 'innovationNum': 4}
{'source': 5, 'dest': 12, 'weight': -0.8061238001243112, 'enabled': True, 'innovationNum': 5}
{'source': 7, 'dest': 12, 'weight': -1.8031790817629423, 'enabled': True, 'innovationNum': 7}
{'source': 8, 'dest': 12, 'weight': 0.06752375424319101, 'enabled': True, 'innovationNum': 8}
{'source': 9, 'dest': 12, 'weight': -0.41210342734771865, 'enabled': True, 'innovationNum': 9}
{'source': 10, 'dest': 12, 'weight': -1.1927998573843281, 'enabled': True, 'innovationNum': 10}
{'source': 0, 'dest': 13, 'weight': 1.5174842204398225, 'enabled': True, 'innovationNum': 11}
{'source': 1, 'dest': 13, 'weight': -0.1365222599237601, 'enabled': True, 'innovationNum': 12}
{'source': 2, 'dest': 13, 'weight': 1.5047264669645806, 'enabled': True, 'innovationNum': 13}
{'source': 3, 'dest': 13, 'weight': -0.3462521774467797, 'enabled': True, 'innovationNum': 14}
{'source': 4, 'dest': 13, 'weight': 1.4033781722835532, 'enabled': True, 'innovationNum': 15}
{'source': 5, 'dest': 13, 'weight': 1.0531228298140496, 'enabled': True, 'innovationNum': 16}
{'source': 7, 'dest': 13, 'weight': 1.069505878205114, 'enabled': True, 'innovationNum': 18}
{'source': 8, 'dest': 13, 'weight': -1.5534370613584296, 'enabled': True, 'innovationNum': 19}
{'source': 0, 'dest': 14, 'weight': -0.44830550997926233, 'enabled': True, 'innovationNum': 22}
{'source': 1, 'dest': 14, 'weight': -1.0015783658189075, 'enabled': True, 'innovationNum': 23}
{'source': 2, 'dest': 14, 'weight': 1.069225019321407, 'enabled': True, 'innovationNum': 24}
{'source': 3, 'dest': 14, 'weight': -0.11486570767696191, 'enabled': True, 'innovationNum': 25}
{'source': 4, 'dest': 14, 'weight': -1.689995878262934, 'enabled': True, 'innovationNum': 26}
{'source': 5, 'dest': 14, 'weight': 0.6197349310759868, 'enabled': True, 'innovationNum': 27}
{'source': 6, 'dest': 14, 'weight': -0.1795434546932726, 'enabled': True, 'innovationNum': 28}
{'source': 7, 'dest': 14, 'weight': 1.854806860697133, 'enabled': True, 'innovationNum': 29}
{'source': 9, 'dest': 14, 'weight': -1.6720917606880956, 'enabled': True, 'innovationNum': 31}
{'source': 10, 'dest': 14, 'weight': 0.056550747468321116, 'enabled': True, 'innovationNum': 32}
{'source': 0, 'dest': 15, 'weight': 0.22173460792411204, 'enabled': True, 'innovationNum': 33}
{'source': 1, 'dest': 15, 'weight': -1.5812298569014134, 'enabled': True, 'innovationNum': 34}
{'source': 3, 'dest': 15, 'weight': -1.1130919125215426, 'enabled': True, 'innovationNum': 36}
{'source': 4, 'dest': 15, 'weight': 0.9371587929218412, 'enabled': True, 'innovationNum': 37}
{'source': 5, 'dest': 15, 'weight': -1.8840682363654997, 'enabled': True, 'innovationNum': 38}
{'source': 6, 'dest': 15, 'weight': 1.7465514372437334, 'enabled': False, 'innovationNum': 39}
{'source': 7, 'dest': 15, 'weight': 1.4843034815983627, 'enabled': True, 'innovationNum': 40}
{'source': 8, 'dest': 15, 'weight': 2, 'enabled': True, 'innovationNum': 41}
{'source': 9, 'dest': 15, 'weight': -1.0948271251423547, 'enabled': True, 'innovationNum': 42}
{'source': 10, 'dest': 15, 'weight': 1.725608203712083, 'enabled': True, 'innovationNum': 43}
{'source': 0, 'dest': 16, 'weight': -1.9935408577359133, 'enabled': True, 'innovationNum': 44}
{'source': 1, 'dest': 16, 'weight': -1.9623379993993062, 'enabled': True, 'innovationNum': 45}
{'source': 2, 'dest': 16, 'weight': 1.0945305456500445, 'enabled': True, 'innovationNum': 46}
{'source': 3, 'dest': 16, 'weight': -0.15845386418173835, 'enabled': True, 'innovationNum': 47}
{'source': 4, 'dest': 16, 'weight': 1.7545480116625738, 'enabled': True, 'innovationNum': 48}
{'source': 7, 'dest': 16, 'weight': -0.26637960589018705, 'enabled': True, 'innovationNum': 51}
{'source': 8, 'dest': 16, 'weight': 0.06825713191972538, 'enabled': True, 'innovationNum': 52}
{'source': 9, 'dest': 16, 'weight': -1.595013680032583, 'enabled': True, 'innovationNum': 53}
{'source': 10, 'dest': 16, 'weight': 0.19631675838665696, 'enabled': True, 'innovationNum': 54}
{'source': 6, 'dest': 16, 'weight': -0.29747435450936566, 'enabled': True, 'innovationNum': 50}
{'source': 5, 'dest': 16, 'weight': 0.3136485160736837, 'enabled': True, 'innovationNum': 49}
{'source': 8, 'dest': 14, 'weight': 0.9526554527791431, 'enabled': True, 'innovationNum': 30}
{'source': 9, 'dest': 13, 'weight': 2, 'enabled': True, 'innovationNum': 20}
{'source': 10, 'dest': 13, 'weight': -1.522393636811124, 'enabled': True, 'innovationNum': 21}
{'source': 2, 'dest': 15, 'weight': 0.9584083753684638, 'enabled': True, 'innovationNum': 35}
{'source': 6, 'dest': 13, 'weight': -1.8433404301567977, 'enabled': True, 'innovationNum': 17}
{'source': 6, 'dest': 12, 'weight': -1.6296801680194646, 'enabled': True, 'innovationNum': 6}
'''

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

def visualize(p):

    edge_data = [[e["source"], e["dest"], str(e["weight"])] for e in p["phenotype"]["edges"]]
    
    edge_df = pd.DataFrame(edge_data, columns=['from', 'to', 'label'])

    node_data = [[n["id"]] for n in p["phenotype"]["nodes"]]
    node_df = pd.DataFrame(node_data, columns=["id"])

    # init Jaal and run server
    Jaal(edge_df, node_df).plot(directed=True)

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

def get_training_data(start_date, end_date):

    # Get the data of the stock
    apple = yf.Ticker(TICKER_SYMBOL)

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

    # Normalize each feature using a rolling window
    for feature in features:
        rolling_mean = historical_prices[feature].rolling(window=20).mean()
        rolling_std = historical_prices[feature].rolling(window=20).std()

        # Normalize the feature (subtract rolling mean, divide by rolling std dev)
        data[f'{feature}'] = (historical_prices[feature] - rolling_mean) / rolling_std
    data["Close_Not_Normalized"] = historical_prices["Close"]

    # for feature in features:
    #     # data[f'{feature}'] = historical_prices[feature] / historical_prices[feature].iloc[0]
    #     data[f'{feature}'] = historical_prices[feature]
    # # data["Price_Change"] = historical_prices["Price_Change"] * 1000

    data.dropna(inplace=True)

    return data.loc[start_date:end_date]

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

price_data = get_training_data(START_DATE, END_DATE)
t = {
    "fitness": 0,
    "held": 0,
    "cash": STARTING_CASH,
    "portfolio_values": [],
    "phenotype": {
        "nodes": nodes,
        "edges": edges
    }
}

history = []

for i, (index, row) in enumerate(price_data.iterrows()):
    inputs = row.tolist() + [float((t["held"] * row["Close_Not_Normalized"]) / (t["held"] * row["Close_Not_Normalized"] + t["cash"]))] # Add amount held as proportion of portfolio value to inputs
    inputs.remove(row["Close_Not_Normalized"])
    decision = produce_move(t["phenotype"], inputs)
    decision_index = decision.index(max(decision))
    if decision_index == 0 or (decision_index == 1 and t["held"] < 10):
        t["cash"] += t["held"] * row["Close_Not_Normalized"]
        t["held"] = 0
    elif decision_index == 1:
        t["cash"] += 10 * row["Close_Not_Normalized"]
        t["held"] -= 10
    elif decision_index == 3 and t["cash"] >= 10 * row["Close_Not_Normalized"]:
        t["cash"] -= 10 * row["Close_Not_Normalized"]
        t["held"] += 10
    elif decision_index == 4 and t["cash"] >= 50 * row["Close_Not_Normalized"]:
        t["cash"] -= 50 * row["Close_Not_Normalized"]
        t["held"] += 50

    t["portfolio_values"].append(t["held"] * row["Close_Not_Normalized"] + t["cash"])

    history.append({
        "Time": i,
        "Close": row["Close_Not_Normalized"],
        "Portfolio_value": t["held"] * row["Close_Not_Normalized"] + t["cash"],
        "Held": t["held"],
        "Decision": decision_index
    })

print("Final fitness: ", t["fitness"])

# Create a new figure and axis
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot the data
ax.plot([h["Time"] for h in history], [h["Close"] for h in history], label='Closing Price')

# for i in [h["Time"] for h in history if h["Decision"] == 0]:
#     ax.axvline(x=i, color='g', linestyle='-.')

ax2.plot([h["Time"] for h in history], [h["Portfolio_value"] for h in history], label='Portfolio Value')

# Customize the plot
ax.set_title('Stock Movement')
ax.legend()

ax2.set_title("Portfolio Value")

# visualize(t)

plt.show()