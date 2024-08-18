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

TICKER_SYMBOL = "GC=F"
START_DATE = "2024-08-16"
END_DATE = "2024-08-16"

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
{'id': 222, 'value': 0, 'layer': 'hidden'}

edges:
{'source': 0, 'dest': 12, 'weight': -0.5833662684757117, 'enabled': True, 'innovationNum': 0}
{'source': 1, 'dest': 12, 'weight': -1.4471862086647793, 'enabled': True, 'innovationNum': 1}
{'source': 2, 'dest': 12, 'weight': -1.1069683981219205, 'enabled': True, 'innovationNum': 2}
{'source': 3, 'dest': 12, 'weight': -0.3791481449982461, 'enabled': True, 'innovationNum': 3}
{'source': 4, 'dest': 12, 'weight': -1.4847916747131982, 'enabled': True, 'innovationNum': 4}
{'source': 5, 'dest': 12, 'weight': -0.22682213561708942, 'enabled': True, 'innovationNum': 5}
{'source': 6, 'dest': 12, 'weight': 0.46877172282716995, 'enabled': True, 'innovationNum': 6}
{'source': 7, 'dest': 12, 'weight': -0.8962371949967172, 'enabled': True, 'innovationNum': 7}
{'source': 8, 'dest': 12, 'weight': 0.02272280420757733, 'enabled': True, 'innovationNum': 8}
{'source': 9, 'dest': 12, 'weight': -1.9382462463395567, 'enabled': True, 'innovationNum': 9}
{'source': 10, 'dest': 12, 'weight': -1.7921440089633505, 'enabled': True, 'innovationNum': 10}
{'source': 0, 'dest': 13, 'weight': 0.7456164777229346, 'enabled': True, 'innovationNum': 11}
{'source': 1, 'dest': 13, 'weight': 0.1549033129881563, 'enabled': True, 'innovationNum': 12}
{'source': 2, 'dest': 13, 'weight': 0.9569334277750197, 'enabled': True, 'innovationNum': 13}
{'source': 3, 'dest': 13, 'weight': 0.9260954128064438, 'enabled': True, 'innovationNum': 14}
{'source': 4, 'dest': 13, 'weight': -0.3594091226880507, 'enabled': True, 'innovationNum': 15}
{'source': 5, 'dest': 13, 'weight': 1.9173748556045682, 'enabled': True, 'innovationNum': 16}
{'source': 6, 'dest': 13, 'weight': -1.2689495778818336, 'enabled': True, 'innovationNum': 17}
{'source': 7, 'dest': 13, 'weight': 1.0987338519563843, 'enabled': True, 'innovationNum': 18}
{'source': 8, 'dest': 13, 'weight': -1.7813788075445665, 'enabled': True, 'innovationNum': 19}
{'source': 0, 'dest': 14, 'weight': 1.2354464985203784, 'enabled': True, 'innovationNum': 22}
{'source': 1, 'dest': 14, 'weight': 1.0026306606459117, 'enabled': True, 'innovationNum': 23}
{'source': 2, 'dest': 14, 'weight': -0.5164627307050176, 'enabled': True, 'innovationNum': 24}
{'source': 3, 'dest': 14, 'weight': 0.27639805281952734, 'enabled': True, 'innovationNum': 25}
{'source': 4, 'dest': 14, 'weight': -0.13168534165489684, 'enabled': True, 'innovationNum': 26}
{'source': 6, 'dest': 14, 'weight': 1.3396484972982292, 'enabled': True, 'innovationNum': 28}
{'source': 7, 'dest': 14, 'weight': -0.9101936301119578, 'enabled': True, 'innovationNum': 29}
{'source': 8, 'dest': 14, 'weight': -0.030195880579670487, 'enabled': True, 'innovationNum': 30}
{'source': 9, 'dest': 14, 'weight': -0.2368852244334012, 'enabled': True, 'innovationNum': 31}
{'source': 10, 'dest': 14, 'weight': 1.279413416149191, 'enabled': True, 'innovationNum': 32}
{'source': 0, 'dest': 15, 'weight': -1.560584476117979, 'enabled': True, 'innovationNum': 33}
{'source': 1, 'dest': 15, 'weight': -1.6128894689813382, 'enabled': True, 'innovationNum': 34}
{'source': 2, 'dest': 15, 'weight': 0.9203773763334113, 'enabled': True, 'innovationNum': 35}
{'source': 3, 'dest': 15, 'weight': -1.1747680019701563, 'enabled': True, 'innovationNum': 36}
{'source': 4, 'dest': 15, 'weight': -0.0925221673098906, 'enabled': True, 'innovationNum': 37}
{'source': 5, 'dest': 15, 'weight': 0.34408602345956396, 'enabled': True, 'innovationNum': 38}
{'source': 6, 'dest': 15, 'weight': 0.6905147621869236, 'enabled': True, 'innovationNum': 39}
{'source': 7, 'dest': 15, 'weight': -0.33632410032778415, 'enabled': True, 'innovationNum': 40}
{'source': 8, 'dest': 15, 'weight': -0.9704200506261862, 'enabled': True, 'innovationNum': 41}
{'source': 9, 'dest': 15, 'weight': 0.339975243136495, 'enabled': True, 'innovationNum': 42}
{'source': 10, 'dest': 15, 'weight': -0.11698804226738928, 'enabled': True, 'innovationNum': 43}
{'source': 0, 'dest': 16, 'weight': 0.7434961273767096, 'enabled': True, 'innovationNum': 44}
{'source': 1, 'dest': 16, 'weight': -0.09241635858878228, 'enabled': False, 'innovationNum': 45}
{'source': 2, 'dest': 16, 'weight': -1.2254735257839058, 'enabled': True, 'innovationNum': 46}
{'source': 3, 'dest': 16, 'weight': -1.4950754830735873, 'enabled': True, 'innovationNum': 47}
{'source': 4, 'dest': 16, 'weight': -1.1636188730453347, 'enabled': True, 'innovationNum': 48}
{'source': 5, 'dest': 16, 'weight': -0.09370480055043325, 'enabled': True, 'innovationNum': 49}
{'source': 6, 'dest': 16, 'weight': 0.6204042376599904, 'enabled': True, 'innovationNum': 50}
{'source': 7, 'dest': 16, 'weight': -0.45157499950335556, 'enabled': True, 'innovationNum': 51}
{'source': 8, 'dest': 16, 'weight': 0.3771773694373679, 'enabled': True, 'innovationNum': 52}
{'source': 9, 'dest': 16, 'weight': 1.8697992498168974, 'enabled': True, 'innovationNum': 53}
{'source': 10, 'dest': 16, 'weight': -0.588212900333533, 'enabled': True, 'innovationNum': 54}
{'source': 9, 'dest': 13, 'weight': -1.576947311548878, 'enabled': True, 'innovationNum': 20}
{'source': 10, 'dest': 13, 'weight': 1.6831157374897754, 'enabled': True, 'innovationNum': 21}
{'source': 5, 'dest': 222, 'weight': 1, 'enabled': True, 'innovationNum': 497}
{'source': 222, 'dest': 14, 'weight': 1.1596455517660575, 'enabled': True, 'innovationNum': 498}
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