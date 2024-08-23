# Note: Can train to a high level on every day, but need to make sure that the same trader is what's doing well on all of them
# - Could do something where when a trader get's a high level they are tested on each day, getting further trained on the day it did worse on until good at all. Maybe keep one day to be test only?
# - Also, fitness is very unnaturally high immediately upon switching to a new day, before going down again, fixed now that I've moved reset_players to earlier? I feel like maybe not
# - How to get rid of local optimums? Sometimes we get stuck and stop improving

# Note: trained one to 18,000 within 20 rounds, and then switched to next day and immediately got 23,000, but got 0 on next day, quickly trained to 2000, then got 4000 on next day, 18000, then 8000 on the next, quickly trained back to 23,000 the next right away
# - Again though, same trader or not? 

# Train on data from a stock which trended down?

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

NUM_INPUT_NODES = 11
NUM_OUTPUT_NODES = 5
GENERATION_POPULATION_LIMIT = 300
SPECIES_LIMIT = 3
GENERATION_KEEP_CONSTANT = 0.16
BABY_MUTATION_CHANCE = 1

MINUTES_TRAINED_PER_DAY = 100000
STARTING_CASH = 1000000
MAX_TRAINING_GENERATIONS = 20
TICKER = "BTC-USD"

global innovationCounter
innovationCounter = 0
globalEdges = []

global nodeCounter
nodeCounter = NUM_OUTPUT_NODES + NUM_INPUT_NODES + 1

global species_similarity
species_similarity = 0.99

def visualize(p):

    edge_data = [[e["source"], e["dest"], str(e["weight"])] for e in p["phenotype"]["edges"]]
    
    edge_df = pd.DataFrame(edge_data, columns=['from', 'to', 'label'])

    node_data = [[n["id"]] for n in p["phenotype"]["nodes"]]
    node_df = pd.DataFrame(node_data, columns=["id"])

    # init Jaal and run server
    Jaal(edge_df, node_df).plot(directed=True)

def sigmoid(x):
    return 1 / (1 + (math.e ** x))

def print_out(p):
    print("nodes:")
    for n in p["phenotype"]["nodes"]:
        print(n)
    print("\nedges:")
    for e in p["phenotype"]["edges"]:
        print(e)

def add_edge(edges, source, dest, weight, enabled = True):

    # If unit already has edge just leave
    if len([e for e in edges if e["source"] == source and e["dest"] == dest]) > 0:
        return

    global innovationCounter

    innovationNum = innovationCounter
    for i in globalEdges:
        if i["source"] == source and i["dest"] == dest:
            innovationNum = i["innovationNum"]

    newEdge = {
        "source": source,
        "dest": dest,
        "weight": weight,
        "enabled": enabled,
        "innovationNum": innovationNum
    }
    
    if innovationNum == innovationCounter:
        globalEdges.append(newEdge)
        innovationCounter += 1

    edges.append(newEdge)

def make_player():

    nodes = []
    edges = []
    i = 0 # Nodes counter
    j = 0 # Edges counter

    # Bias input node
    nodes.append({
        "id": i,
        "value": 1,
        "layer": "input"
    })
    i += 1

    # The rest of the input nodes
    for l in range(NUM_INPUT_NODES):
        nodes.append({
            "id": i,
            "value": 0,
            "layer": "input"
        })
        i += 1

    # The output nodes
    for l in range(NUM_OUTPUT_NODES):
        nodes.append({
            "id": i,
            "value": 0,
            "layer": "output"
        })
        # Default random edges from each input to each output
        for n in range(NUM_INPUT_NODES):
            add_edge(edges, n, i, random.uniform(-2, 2))
        # add_edge(edges, 0, i, random.uniform(-2, 2))
        i += 1

    return {
        "nodes": nodes,
        "edges": edges
    }

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


def cross_over(p1, p2):

    new_edges = []

    shared_edges = [[e["source"], e["dest"]] for e in p1["phenotype"]["edges"] if [e["source"], e["dest"]] in [[f["source"], f["dest"]] for f in p2["phenotype"]["edges"]]]
    disjoint_p1 = [[e["source"], e["dest"]] for e in p1["phenotype"]["edges"] if [e["source"], e["dest"]] not in shared_edges]
    disjoint_p2 = [[e["source"], e["dest"]] for e in p2["phenotype"]["edges"] if [e["source"], e["dest"]] not in shared_edges]

    for shared in shared_edges:
        copied_edge = random.choice([e for e in p1["phenotype"]["edges"] + p2["phenotype"]["edges"] if e["source"] == shared[0] and e["dest"] == shared[1]])
        add_edge(new_edges, shared[0], shared[1], copied_edge["weight"], copied_edge["enabled"])

    disjoint = []
    if p1["fitness"] >= p2["fitness"]:
        disjoint = disjoint_p1
    else: 
        disjoint = disjoint_p2

    for d in disjoint:
        copied_edge = random.choice([e for e in p1["phenotype"]["edges"] + p2["phenotype"]["edges"] if e["source"] == d[0] and e["dest"] == d[1]])
        add_edge(new_edges, d[0], d[1], copied_edge["weight"], copied_edge["enabled"])

    nodes = []
    for node in p1["phenotype"]["nodes"] + p2["phenotype"]["nodes"]:
        if node["id"] in [e["source"] for e in new_edges] + [e["dest"] for e in new_edges] and node not in nodes:
            nodes.append(node)

    for id in range(NUM_INPUT_NODES):
        if id not in [n["id"] for n in nodes]:
            nodes.append({
                "id": id,
                "value": 0,
                "layer": "input"
            })

    return make_trader({
        "nodes": nodes,
        "edges": new_edges
    })

def cycle(p, source, dest):
    edges = []
    nodes = {}  # Use a dictionary for faster lookup

    for e in p["phenotype"]["edges"]:
        edges.append([e["source"], e["dest"]])

    edges.append([source["id"], dest["id"]])

    for n in p["phenotype"]["nodes"]:
        nodes[n["id"]] = False  # False means not visited

    def cycle_recursive(node_id, visited, recursion_stack):
        visited[node_id] = True
        recursion_stack[node_id] = True

        for neighbor_id in [edge[1] for edge in edges if edge[0] == node_id]:
            if not visited[neighbor_id]:
                if cycle_recursive(neighbor_id, visited, recursion_stack):
                    return True
            elif recursion_stack[neighbor_id]:
                return True

        recursion_stack[node_id] = False
        return False

    visited = {node_id: False for node_id in nodes}
    recursion_stack = {node_id: False for node_id in nodes}

    for node_id in nodes:
        if not visited[node_id]:
            if cycle_recursive(node_id, visited, recursion_stack):
                return True

    return False


def mutate_node(p):

    global nodeCounter

    split_edge = random.choice([e for e in p["phenotype"]["edges"]])
    source_node = [n for n in p["phenotype"]["nodes"] if n["id"] == split_edge["source"]][0]
    destination_node = [n for n in p["phenotype"]["nodes"] if n["id"] == split_edge["dest"]][0]

    new_node = {
        "id": nodeCounter, 
        "value": 0,
        "layer": "hidden"
    }
    nodeCounter += 1

    add_edge(p["phenotype"]["edges"], source_node["id"], new_node["id"], 1)
    add_edge(p["phenotype"]["edges"], new_node["id"], destination_node["id"], split_edge["weight"], split_edge["enabled"])

    p["phenotype"]["edges"].remove(split_edge)
    p["phenotype"]["nodes"].append(new_node)


def mutate_edge(p):
    counter = 0
    while True:
        source = random.choice([n for n in p["phenotype"]["nodes"] if n["layer"] != "output"])
        dest = random.choice([n for n in p["phenotype"]["nodes"] if n["layer"] != "input"])
        counter += 1
        if not cycle(p, source, dest) and source["id"] != dest["id"]:
            break
        if counter > 10:
            return

    add_edge(p["phenotype"]["edges"], source["id"], dest["id"], random.uniform(-2, 2))
    

def mutate_enable_disable(p):
    edge = random.choice([e for e in p["phenotype"]["edges"]])
    if edge["enabled"] == True:
        edge["enabled"] = False
    else:
        edge["enabled"] = True


def mutate_weight_shift(p):
    edge = random.choice([e for e in p["phenotype"]["edges"]])
    edge["weight"] *= random.uniform(0, 2)
    if edge["weight"] > 2:
        edge["weight"] = 2
    elif edge["weight"] < -2:
        edge["weight"] = -2


def mutate_weight_random(p):
    edge = random.choice([e for e in p["phenotype"]["edges"]])
    edge["weight"] = random.uniform(-2, 2)

# This distribution is random
def mutate(p):
    mutation_options = [
        mutate_weight_shift,
        mutate_weight_random,
        mutate_edge,
        mutate_node,
        mutate_enable_disable
    ]
    mutation_weights = [0.5, 0.2, 0.2, 0.05, 0.05] 

    chosen_mutation = random.choices(mutation_options, weights=mutation_weights, k=1)[0]
    chosen_mutation(p)

# randomly deciding that if two players both have similar edges than they are same species
def sameSpecies(p1, p2):

    shared_edges = [[e["source"], e["dest"]] for e in p1["phenotype"]["edges"] if [e["source"], e["dest"]] in [[f["source"], f["dest"]] for f in p2["phenotype"]["edges"]]]
    if len(shared_edges) / len(p1["phenotype"]["edges"]) >= species_similarity and len(shared_edges) / len(p2["phenotype"]["edges"]) >= species_similarity:
        return True
    else:
        return False


def speciation(players):
    species = []

    for p in players:
        found_species = False
        for s in species:
            if sameSpecies(p, s[0]) and not found_species:
                s.append(p)
                found_species = True

        if not found_species:
            species.append([p])

    if len(species) > SPECIES_LIMIT:
        global species_similarity
        species_similarity = species_similarity - 0.1
        return speciation(players)

    return species

def next_generation(players):

    next_gen = []

    # Speciate and eliminate bottom 50% of each species
    species = speciation(players)
    # print("\n")
    for i, s in enumerate(species):
        sorted_s = sorted(s, key=lambda x: x['fitness'], reverse=True)

        keep = 0.5
        if len(s) > 8:
            keep =  GENERATION_KEEP_CONSTANT
        # print(f"\nSpecies {i} top fitness: {sorted_s[0]['fitness']}, average fitness: {sum([s['fitness'] for s in sorted_s]) // len(sorted_s)} number of players: {len(sorted_s)}")
        for i in range(int(len(sorted_s) * keep) + 1):
            next_gen.append(sorted_s[i])

    species = speciation(next_gen)

    # Refill to generation limit by randomly selecting players to breed
    while len(next_gen) < GENERATION_POPULATION_LIMIT:
        s = random.choice(species)
        mommy = random.choice(s)
        daddy = random.choice(s)
        baby = cross_over(mommy, daddy)
        if random.randint(1,BABY_MUTATION_CHANCE) == 1: #chance of mutating baby after cross-over
            mutate(baby)
        next_gen.append(baby)
        s.append(baby)

    return next_gen

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
    apple = yf.Ticker(TICKER)

    # Get the historical prices for Apple stock
    historical_prices = apple.history(period='max', interval='1m')
    # del historical_prices["Dividends"]
    # del historical_prices["Stock Splits"]

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

def make_trader(phenotype):
    return {
        "fitness": 0,
        "held": 0,
        "cash": STARTING_CASH,
        "portfolio_values": [],
        "phenotype": phenotype
    }

def reset_traders(traders):
    for t in traders:
        t["fitness"] = 0
        t["held"] = 0
        t["cash"] = STARTING_CASH
        t["portfolio_values"] = []

def calculate_trader_fitness(traders):
    for i, t in enumerate(traders):
        returns = [t["portfolio_values"][i] - t["portfolio_values"][i - 1] for i in range(1, len(t["portfolio_values"]))]
        mean_return = np.mean(returns)

        # Old code for the Sharpe ratio
        # standard_deviation = np.std(returns)
        # if standard_deviation == 0:
        #     t["fitnesss"] = 0
        #     continue

        # Calculate downside deviation
        downside_returns = [r for r in returns if r < 0]
        if len(downside_returns) > 0:
            downside_deviation = np.std(downside_returns)
        else:
            downside_deviation = 0

        if downside_deviation == 0:
            t["fitness"] = 0
        else:
            t["fitness"] = (mean_return) / downside_deviation

        # t["fitness"] = mean_return / standard_deviation # Sharpe ratio for fitness function

traders = [make_trader(make_player()) for _ in range(GENERATION_POPULATION_LIMIT)]
runs = 0
date_counter = 0

for generation_counter in range(MAX_TRAINING_GENERATIONS):

    try:

        reset_traders(traders)

        # if 8 + (date_counter % 6) >= 10:
        #     date = f"2024-08-{8 + (date_counter % 6)}"
        # else:
        #     date = f"2024-08-0{8 + (date_counter % 6)}"
        date = "2024-08-14"
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        training_data = get_training_data("2024-08-08", "2024-08-19")

        if len(training_data) == 0:
            date_counter += 1
            continue

        minute_counter = 0

        # with alive_bar(len(get_training_data(date, date)), title="Simulating Trading Day...") as bar:
        with alive_bar(min(MINUTES_TRAINED_PER_DAY, len(training_data)), title="Simulating Trading Day...") as bar:
        
            for index, row in training_data.iterrows():

                minute_counter += 1
                if minute_counter > MINUTES_TRAINED_PER_DAY:
                    break

                for i, t in enumerate(traders):
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

                bar()

        runs += 1

        calculate_trader_fitness(traders)

        sorted_traders = sorted(traders, key=lambda x: x['fitness'], reverse=True)
        print(f"Run: {runs}, Date: {date}, Maximum fitness: {sorted_traders[0]['fitness']}, average fitness: {sum([s['fitness'] for s in sorted_traders]) / len(sorted_traders)}, number of species: {len(speciation(traders))}\n")
        traders = next_generation(traders)

    except KeyboardInterrupt:
        x  = int(input("\nWhat would you like to do?\n1: Move to next date.\n2: Print out currently winning trader.\n3: Quit.\n\nInput: "))
        if x == 1:
            date_counter += 1
            continue
        elif x == 2:
            print_out(sorted_traders[0])
            continue
        else:
            quit()

sorted_traders = sorted(traders, key=lambda x: x['fitness'], reverse=True)
print_out(sorted_traders[0])