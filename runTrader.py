import random
import math
from collections import deque

import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from alive_progress import alive_bar

TICKER_SYMBOL = "BTC-USD"
START_DATE = "2024-08-23"
END_DATE = "2024-08-23"

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
{'source': 0, 'dest': 12, 'weight': 0.04901413550845346, 'enabled': True, 'innovationNum': 0}
{'source': 1, 'dest': 12, 'weight': -1.6302670785006748, 'enabled': True, 'innovationNum': 1}
{'source': 2, 'dest': 12, 'weight': -0.402426777573512, 'enabled': True, 'innovationNum': 2}
{'source': 3, 'dest': 12, 'weight': -1.3903952350832833, 'enabled': True, 'innovationNum': 3}
{'source': 4, 'dest': 12, 'weight': 0.2539284637841077, 'enabled': True, 'innovationNum': 4}
{'source': 5, 'dest': 12, 'weight': -1.1244523278139424, 'enabled': True, 'innovationNum': 5}
{'source': 6, 'dest': 12, 'weight': -1.598541022803401, 'enabled': True, 'innovationNum': 6}
{'source': 7, 'dest': 12, 'weight': 1.352997942935004, 'enabled': True, 'innovationNum': 7}
{'source': 8, 'dest': 12, 'weight': 0.5167040641615919, 'enabled': True, 'innovationNum': 8}
{'source': 9, 'dest': 12, 'weight': 1.933903095993132, 'enabled': True, 'innovationNum': 9}
{'source': 10, 'dest': 12, 'weight': -0.47231657303134966, 'enabled': True, 'innovationNum': 10}
{'source': 0, 'dest': 13, 'weight': -1.3956468681265495, 'enabled': True, 'innovationNum': 11}
{'source': 1, 'dest': 13, 'weight': -1.3201762566504724, 'enabled': True, 'innovationNum': 12}
{'source': 2, 'dest': 13, 'weight': 0.7205737761119231, 'enabled': True, 'innovationNum': 13}
{'source': 3, 'dest': 13, 'weight': 1.7400454810817885, 'enabled': True, 'innovationNum': 14}
{'source': 4, 'dest': 13, 'weight': 2, 'enabled': True, 'innovationNum': 15}
{'source': 5, 'dest': 13, 'weight': 1.8235885862334045, 'enabled': True, 'innovationNum': 16}
{'source': 6, 'dest': 13, 'weight': -1.9346439178666905, 'enabled': True, 'innovationNum': 17}
{'source': 7, 'dest': 13, 'weight': -1.2932955265291874, 'enabled': True, 'innovationNum': 18}
{'source': 8, 'dest': 13, 'weight': -1.8936788417335015, 'enabled': True, 'innovationNum': 19}
{'source': 9, 'dest': 13, 'weight': 1.164086281894563, 'enabled': True, 'innovationNum': 20}
{'source': 10, 'dest': 13, 'weight': -1.390823571702342, 'enabled': True, 'innovationNum': 21}
{'source': 0, 'dest': 14, 'weight': -0.3286338747064179, 'enabled': True, 'innovationNum': 22}
{'source': 1, 'dest': 14, 'weight': -1.9238780573868466, 'enabled': True, 'innovationNum': 23}
{'source': 2, 'dest': 14, 'weight': 0.2689363113998051, 'enabled': True, 'innovationNum': 24}
{'source': 3, 'dest': 14, 'weight': -0.13300561543669787, 'enabled': True, 'innovationNum': 25}
{'source': 4, 'dest': 14, 'weight': -0.3639477403662439, 'enabled': True, 'innovationNum': 26}
{'source': 5, 'dest': 14, 'weight': -0.9283459124556521, 'enabled': True, 'innovationNum': 27}
{'source': 6, 'dest': 14, 'weight': -1.4107993754908894, 'enabled': True, 'innovationNum': 28}
{'source': 7, 'dest': 14, 'weight': -1.0129721037541435, 'enabled': True, 'innovationNum': 29}
{'source': 8, 'dest': 14, 'weight': -1.6848511669899509, 'enabled': True, 'innovationNum': 30}
{'source': 9, 'dest': 14, 'weight': 0.14521062947728813, 'enabled': True, 'innovationNum': 31}
{'source': 10, 'dest': 14, 'weight': 0.7690853322292197, 'enabled': True, 'innovationNum': 32}
{'source': 0, 'dest': 15, 'weight': 1.1844675298670877, 'enabled': True, 'innovationNum': 33}
{'source': 1, 'dest': 15, 'weight': 0.08233374117619813, 'enabled': True, 'innovationNum': 34}
{'source': 2, 'dest': 15, 'weight': -0.3244248867398678, 'enabled': True, 'innovationNum': 35}
{'source': 4, 'dest': 15, 'weight': 0.06005808209695607, 'enabled': True, 'innovationNum': 37}
{'source': 5, 'dest': 15, 'weight': -0.6904086317934235, 'enabled': True, 'innovationNum': 38}
{'source': 6, 'dest': 15, 'weight': 1.9495710608632049, 'enabled': True, 'innovationNum': 39}
{'source': 7, 'dest': 15, 'weight': 1.8486724349485715, 'enabled': True, 'innovationNum': 40}
{'source': 8, 'dest': 15, 'weight': 0.5107796934759299, 'enabled': True, 'innovationNum': 41}
{'source': 9, 'dest': 15, 'weight': 1.5882891615364163, 'enabled': True, 'innovationNum': 42}
{'source': 10, 'dest': 15, 'weight': 1.714450113512299, 'enabled': True, 'innovationNum': 43}
{'source': 0, 'dest': 16, 'weight': -1.6396500314641096, 'enabled': True, 'innovationNum': 44}
{'source': 1, 'dest': 16, 'weight': -1.6404276361750516, 'enabled': True, 'innovationNum': 45}
{'source': 3, 'dest': 16, 'weight': 0.32732855843248876, 'enabled': True, 'innovationNum': 47}
{'source': 4, 'dest': 16, 'weight': -0.17653851613409577, 'enabled': True, 'innovationNum': 48}
{'source': 5, 'dest': 16, 'weight': 0.8829079101326109, 'enabled': True, 'innovationNum': 49}
{'source': 6, 'dest': 16, 'weight': 1.6135213154674775, 'enabled': False, 'innovationNum': 50}
{'source': 7, 'dest': 16, 'weight': 1.0342737186869675, 'enabled': True, 'innovationNum': 51}
{'source': 8, 'dest': 16, 'weight': 0.06505799427504488, 'enabled': True, 'innovationNum': 52}
{'source': 9, 'dest': 16, 'weight': -0.5232775927168967, 'enabled': True, 'innovationNum': 53}
{'source': 10, 'dest': 16, 'weight': 0.297213728736609, 'enabled': True, 'innovationNum': 54}
{'source': 3, 'dest': 15, 'weight': -1.7357134523025177, 'enabled': True, 'innovationNum': 36}
{'source': 2, 'dest': 16, 'weight': 0.8608205284450823, 'enabled': True, 'innovationNum': 46}
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