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

TICKER_SYMBOL = "AAPL"
START_DATE = "2024-08-14"
END_DATE = "2024-08-14"

# Note: Sells constantly with NVDA, sells constantly after massive price surge with SBUX. Really liked TSLA, many it just sold constantly the whole time. Consistently lost only a little or made a lot
# Note: -4 million on the 12th since it never sold by the end of the day. There was a massive surge in price on the morning of the 13th!

trader_string = '''
nodes:
{'id': 0, 'value': 1, 'layer': 'input'}
{'id': 1, 'value': 0, 'layer': 'input'}
{'id': 2, 'value': 0, 'layer': 'input'}
{'id': 3, 'value': 0, 'layer': 'input'}
{'id': 4, 'value': 0, 'layer': 'input'}
{'id': 5, 'value': 0, 'layer': 'input'}
{'id': 6, 'value': 0, 'layer': 'input'}
{'id': 8, 'value': 0, 'layer': 'output'}
{'id': 9, 'value': 0, 'layer': 'output'}
{'id': 10, 'value': 0, 'layer': 'output'}
{'id': 11, 'value': 0, 'layer': 'output'}
{'id': 12, 'value': 0, 'layer': 'output'}
{'id': 465, 'value': 0, 'layer': 'hidden'}

edges:
{'source': 0, 'dest': 8, 'weight': 0.4695737702264422, 'enabled': False, 'innovationNum': 0}
{'source': 1, 'dest': 8, 'weight': -0.1607716444159215, 'enabled': True, 'innovationNum': 1}
{'source': 2, 'dest': 8, 'weight': 0.1897386768990188, 'enabled': True, 'innovationNum': 2}
{'source': 3, 'dest': 8, 'weight': 0.6291227214508939, 'enabled': True, 'innovationNum': 3}
{'source': 6, 'dest': 8, 'weight': 0.4328785992970223, 'enabled': True, 'innovationNum': 6}
{'source': 0, 'dest': 9, 'weight': 0.0006048469710730173, 'enabled': False, 'innovationNum': 7}
{'source': 1, 'dest': 9, 'weight': -1.0868126651227046, 'enabled': True, 'innovationNum': 8}
{'source': 3, 'dest': 9, 'weight': 1.4736983911765398, 'enabled': True, 'innovationNum': 10}
{'source': 4, 'dest': 9, 'weight': -0.21368972627941632, 'enabled': True, 'innovationNum': 11}
{'source': 5, 'dest': 9, 'weight': -1.8133324576111516, 'enabled': True, 'innovationNum': 12}
{'source': 6, 'dest': 9, 'weight': 1.0501364565959412, 'enabled': True, 'innovationNum': 13}
{'source': 0, 'dest': 10, 'weight': 0.3831012685230819, 'enabled': True, 'innovationNum': 14}
{'source': 2, 'dest': 10, 'weight': -0.526172786923572, 'enabled': True, 'innovationNum': 16}
{'source': 3, 'dest': 10, 'weight': 1.592864133954349, 'enabled': True, 'innovationNum': 17}
{'source': 4, 'dest': 10, 'weight': 1.3098786568145542, 'enabled': True, 'innovationNum': 18}
{'source': 5, 'dest': 10, 'weight': -1.4650070662553565, 'enabled': True, 'innovationNum': 19}
{'source': 6, 'dest': 10, 'weight': 1.0365903231766418, 'enabled': True, 'innovationNum': 20}
{'source': 0, 'dest': 11, 'weight': -1.493932924472094, 'enabled': True, 'innovationNum': 21}
{'source': 2, 'dest': 11, 'weight': -1.4085589171971868, 'enabled': True, 'innovationNum': 23}
{'source': 4, 'dest': 11, 'weight': -1.0098705076645587, 'enabled': True, 'innovationNum': 25}
{'source': 6, 'dest': 11, 'weight': -1.1925858542062464, 'enabled': True, 'innovationNum': 27}
{'source': 0, 'dest': 12, 'weight': -1.3064672181551638, 'enabled': True, 'innovationNum': 28}
{'source': 1, 'dest': 12, 'weight': 1.8942800646977553, 'enabled': True, 'innovationNum': 29}
{'source': 2, 'dest': 12, 'weight': 0.18838423910513535, 'enabled': True, 'innovationNum': 30}
{'source': 3, 'dest': 12, 'weight': -0.09476132406329316, 'enabled': True, 'innovationNum': 31}
{'source': 4, 'dest': 12, 'weight': -0.07270458363450727, 'enabled': True, 'innovationNum': 32}
{'source': 5, 'dest': 12, 'weight': 0.2689410000728656, 'enabled': True, 'innovationNum': 33}
{'source': 6, 'dest': 12, 'weight': -0.03529932698647726, 'enabled': True, 'innovationNum': 34}
{'source': 2, 'dest': 9, 'weight': -0.30909704601464005, 'enabled': True, 'innovationNum': 9}
{'source': 5, 'dest': 8, 'weight': -0.6361907844545551, 'enabled': True, 'innovationNum': 5}
{'source': 1, 'dest': 10, 'weight': -1.0258808354511877, 'enabled': True, 'innovationNum': 15}
{'source': 3, 'dest': 11, 'weight': -1.355574047542281, 'enabled': True, 'innovationNum': 24}
{'source': 4, 'dest': 8, 'weight': -0.6943444734218196, 'enabled': True, 'innovationNum': 4}
{'source': 1, 'dest': 11, 'weight': 0.6928694682472933, 'enabled': True, 'innovationNum': 22}
{'source': 5, 'dest': 465, 'weight': 1, 'enabled': True, 'innovationNum': 1078}
{'source': 465, 'dest': 11, 'weight': -0.910940088543986, 'enabled': True, 'innovationNum': 1079}
{'source': 465, 'dest': 12, 'weight': 1.9781094324378334, 'enabled': True, 'innovationNum': 1339}
'''

NUM_INPUT_NODES = 7

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
    ticker = yf.Ticker(TICKER_SYMBOL)

    # Get the historical prices for Apple stock
    historical_prices = ticker.history(period='max', interval='1m')
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
        data[f'{feature}'] = historical_prices[feature]

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
    "phenotype": {
        "nodes": nodes,
        "edges": edges
    }
}

history = []

for i, (index, row) in enumerate(price_data.iterrows()):
    inputs = row.tolist()
    decision = produce_move(t["phenotype"], inputs)
    decision_index = decision.index(max(decision))

    if decision_index == 0 or (decision_index == 1 and t["held"] < 10):
        t["fitness"] += t["held"] * row["Close"]
        t["held"] = 0
    elif decision_index == 1:
        t["fitness"] += 10 * row["Close"]
        t["held"] -= 10
    elif decision_index == 3:
        t["fitness"] -= 10 * row["Close"]
        t["held"] += 10
    elif decision_index == 4:
        t["fitness"] -= 50 * row["Close"]
        t["held"] += 50

    history.append({
        "Time": i,
        "Close": row["Close"],
        "VWAP": row["VWAP"],
        "BB_Upper": row["BB_Upper"],
        "Profit/Loss": t["fitness"],
        "Held": t["held"],
        "Decision": decision_index
    })

print("Final fitness: ", t["fitness"])

# Create a new figure and axis
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot the data
ax.plot([h["Time"] for h in history], [h["Close"] for h in history], label='Closing Price')
ax.plot([h["Time"] for h in history], [h["VWAP"] for h in history], label='VWAP')
ax.plot([h["Time"] for h in history], [h["BB_Upper"] for h in history], label='BB_Upper')

for i in [h["Time"] for h in history if h["Decision"] == 0]:
    ax.axvline(x=i, color='g', linestyle='-.')

ax2.plot([h["Time"] for h in history], [h["Profit/Loss"] for h in history], label='Fitness')

# Customize the plot
ax.set_title('Stock Movement')
ax.legend()

ax2.set_title("Profit/Loss")

# visualize(t)

plt.show()