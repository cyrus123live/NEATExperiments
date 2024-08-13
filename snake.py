# Notes:
# - I just normalized inputs and not sure if that helped or not, haven't trained a good one since
# - Another change I'm not sure about: speciation limits, I think fewer species better, got a good one with zero speciation but that was before change above too

import pygame
import random
import math
from collections import deque

from jaal import Jaal
from jaal.datasets import load_got
import pandas as pd

WIDTH = 600
HEIGHT = 600
PSIZE = 8

NUM_INPUT_NODES = 6
NUM_OUTPUT_NODES = 3
GENERATION_POPULATION_LIMIT = 100
SPECIES_LIMIT = 1
GENERATION_KEEP_CONSTANT = 0.16

pwidth = WIDTH // PSIZE
pheight = HEIGHT // PSIZE

FRAMES_UNTIL_STARVATION = 600
RUNS_PER_DRAWN = 1

global innovationCounter
innovationCounter = 0
globalEdges = []

global nodeCounter
nodeCounter = NUM_OUTPUT_NODES + NUM_INPUT_NODES + 1

global species_similarity
species_similarity = 0.8

play = True

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

def addEdge(edges, source, dest, weight, enabled = True):

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

def makePlayer():

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
            addEdge(edges, n, i, random.uniform(-2, 2))
        # addEdge(edges, 0, i, random.uniform(-2, 2))
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
        addEdge(new_edges, shared[0], shared[1], copied_edge["weight"], copied_edge["enabled"])

    disjoint = []
    if p1["fitness"] >= p2["fitness"]:
        disjoint = disjoint_p1
    else: 
        disjoint = disjoint_p2

    for d in disjoint:
        copied_edge = random.choice([e for e in p1["phenotype"]["edges"] + p2["phenotype"]["edges"] if e["source"] == d[0] and e["dest"] == d[1]])
        addEdge(new_edges, d[0], d[1], copied_edge["weight"], copied_edge["enabled"])

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

    # note: change this to be more variable
    return {
        "positions": [[pwidth - i * PSIZE, pheight // 2] for i in range(10)],
        "every_discovered_position": [],
        "fruit_position": generate_fruit(),
        "direction": "right",
        "frames_since_last_ate": 0,
        "dead": False,
        "fitness": 0,
        "color": "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]),
        "phenotype": {
            "nodes": nodes,
            "edges": new_edges
        }
    }

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

    addEdge(p["phenotype"]["edges"], source_node["id"], new_node["id"], 1)
    addEdge(p["phenotype"]["edges"], new_node["id"], destination_node["id"], split_edge["weight"], split_edge["enabled"])

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

    addEdge(p["phenotype"]["edges"], source["id"], dest["id"], random.uniform(-2, 2))
    

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

    # rand = random.randint(0, 4)
    # if 0 <= rand <= 40:
    #     mutate_weight_random(p)
    # elif 41 <= rand <= 80:
    #     mutate_weight_shift(p)
    # elif 81 <= rand <= 85:
    #     mutate_edge(p)
    # elif 86 <= rand <= 90:
    #     mutate_node(p)
    # else:
    #     mutate_enable_disable(p)

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
        if random.randint(0,1) == 1: #1/2 chance of mutating baby after cross-over
            mutate(baby)
        next_gen.append(baby)
        s.append(baby)

    return next_gen

def generate_fruit():
    return [random.randrange(1, pwidth - 1), random.randrange(1, pheight - 1)]

def get_snake_move(p):

    fruit = p["fruit_position"]
    head = p["positions"][0]

    distance_straight = distance_right = distance_left = 0
    distance_straight_wall = distance_right_wall = distance_left_wall = 0

    # if p["direction"] == "right":
    #     distance_straight = fruit[0] - head[0] / pwidth
    #     distance_left = head[1] - fruit[1] / pheight
    #     distance_right = fruit[1] - head[1] / pheight

    #     distance_straight_wall = pwidth - head[0] / pwidth
    #     distance_left_wall = head[1] / pheight
    #     distance_right_wall = pheight - head[1] / pheight
    # elif p["direction"] == "left":
    #     distance_straight = head[0] - fruit[0] / pwidth
    #     distance_left = fruit[1] - head[1] / pheight
    #     distance_right = head[1] - fruit[1] / pheight

    #     distance_straight_wall = head[0] / pwidth
    #     distance_left_wall = pheight - head[1]
    #     distance_right_wall = head[1] / pheight
    # elif p["direction"] == "up":
    #     distance_straight = head[1] - fruit[1]
    #     distance_left = head[0] - fruit[0] / pwidth
    #     distance_right = fruit[0] - head[0] / pwidth

    #     distance_straight_wall = head[1]
    #     distance_left_wall = head[0] / pwidth
    #     distance_right_wall = pwidth - head[0] / pwidth
    # elif p["direction"] == "down":
    #     distance_straight = fruit[1] - head[1]
    #     distance_left = fruit[0] - head[0] / pwidth
    #     distance_right = head[0] - fruit[0] / pwidth

    #     distance_straight_wall = pheight - head[1] / pheight
    #     distance_left_wall = pwidth - head[0] / pwidth
    #     distance_right_wall = head[0] / pwidth
    if p["direction"] == "right":
        distance_straight = fruit[0] - head[0]
        distance_left = head[1] - fruit[1]
        distance_right = fruit[1] - head[1]

        distance_straight_wall = pwidth - head[0]
        distance_left_wall = head[1]
        distance_right_wall = pheight - head[1]
    elif p["direction"] == "left":
        distance_straight = head[0] - fruit[0]
        distance_left = fruit[1] - head[1]
        distance_right = head[1] - fruit[1]

        distance_straight_wall = head[0]
        distance_left_wall = pheight - head[1]
        distance_right_wall = head[1]
    elif p["direction"] == "up":
        distance_straight = head[1] - fruit[1]
        distance_left = head[0] - fruit[0]
        distance_right = fruit[0] - head[0]

        distance_straight_wall = head[1]
        distance_left_wall = head[0]
        distance_right_wall = pwidth - head[0]
    elif p["direction"] == "down":
        distance_straight = fruit[1] - head[1]
        distance_left = fruit[0] - head[0]
        distance_right = head[0] - fruit[0]

        distance_straight_wall = pheight - head[1]
        distance_left_wall = pwidth - head[0]
        distance_right_wall = head[0]

    desired = produce_move(p["phenotype"], [distance_left, distance_right, distance_straight, distance_left_wall, distance_right_wall, distance_straight_wall])

    index = desired.index(max(desired))
    if index == 0: # continue straight
        if p["direction"] == "right":
            p["direction"] = "right"
        elif p["direction"] == "left":
            p["direction"] = "left"
        elif p["direction"] == "up":
            p["direction"] = "up"
        elif p["direction"] == "down":
            p["direction"] = "down"
    elif index == 1: # turn right
        if p["direction"] == "right":
            p["direction"] = "down"
        elif p["direction"] == "left":
            p["direction"] = "up"
        elif p["direction"] == "up":
            p["direction"] = "right"
        elif p["direction"] == "down":
            p["direction"] = "left"
    elif index == 2: # turn left
        if p["direction"] == "right":
            p["direction"] = "up"
        elif p["direction"] == "left":
            p["direction"] = "down"
        elif p["direction"] == "up":
            p["direction"] = "left"
        elif p["direction"] == "down":
            p["direction"] = "right"
    
if play:

    players = [{
        "positions": [[pwidth - i * PSIZE, pheight // 2] for i in range(10)],
        "every_discovered_position": [],
        "fruit_position": generate_fruit(),
        "direction": "right",
        "frames_since_last_ate": 0,
        "dead": False,
        "fitness": 0,
        "color": "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]),
        "phenotype": s
    } for s in [makePlayer() for i in range(GENERATION_POPULATION_LIMIT)]]

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    running = True

    runs = 0

while play:

    for p in players:
        p["dead"] = False
        p["fitness"] = 0

    runs += 1
    frames = 0

    if runs % RUNS_PER_DRAWN == 0:
        draw = True
    else:   
        draw = False

    while running:

        frames += 1

        screen.fill("#000000")
        move = ""

        # Event handling loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == 27:
                    running = False

        alive_players = [p for p in players if p["dead"] == False]

        for p in alive_players:

            get_snake_move(p)

            p["every_discovered_position"].append(p["positions"][0])

            if p["direction"] == "up":
                p["positions"].insert(0, [p["positions"][0][0], p["positions"][0][1] - 1])
            elif p["direction"] == "down":
                p["positions"].insert(0, [p["positions"][0][0], p["positions"][0][1] + 1])
            elif p["direction"] == "left":
                p["positions"].insert(0, [p["positions"][0][0] - 1, p["positions"][0][1]])
            elif p["direction"] == "right":
                p["positions"].insert(0, [p["positions"][0][0] + 1, p["positions"][0][1]])

            p["positions"].pop(len(p["positions"]) - 1)

            # Reward snake for new territory discovered, up to 99 points, vs 100 for fruit eaten
            if p["positions"][0] not in p["every_discovered_position"] and p["fitness"] % 100 != 99:
                p["fitness"] += 1

            # No fruit collision
            if p["positions"][0] != p["fruit_position"]:
                p["frames_since_last_ate"] += 1

            # Fruit collision
            else:
                p["fitness"] += 100 * ((FRAMES_UNTIL_STARVATION - 10 - p["frames_since_last_ate"]) / (FRAMES_UNTIL_STARVATION - 10))
                p["frames_since_last_ate"] = 0
                p["fruit_position"] = generate_fruit()

            for i in range(len(p["positions"])):
                pygame.draw.rect(screen, p["color"], pygame.Rect(p["positions"][i][0] * PSIZE, p["positions"][i][1] * PSIZE, PSIZE, PSIZE))
            pygame.draw.rect(screen, p["color"], pygame.Rect(p["fruit_position"][0] * PSIZE, p["fruit_position"][1] * PSIZE, PSIZE, PSIZE))

            # Calculate if the snake died
            if p["frames_since_last_ate"] >= FRAMES_UNTIL_STARVATION:
                p["dead"] = True
            if p["positions"][0][0] < 0 or p["positions"][0][0] > pwidth or p["positions"][0][1] < 0 or p["positions"][0][1] > pheight:
                p["dead"] = True

        if len(alive_players) == 0:
            break

        if draw:
            pygame.display.flip()
            clock.tick(120)

        if frames >= 100000: 
            print("winner!")
            draw = True
            continue

    sorted_players = sorted(players, key=lambda x: x['fitness'], reverse=True)
    print(f"Run: {runs}, Maximum fitness: {sorted_players[0]['fitness']}, average fitness: {sum([s['fitness'] for s in sorted_players]) // len(sorted_players)}, number of species: {len(speciation(players))}")

    players = next_generation(players)


