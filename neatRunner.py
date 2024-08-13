import pygame
import random
import math
from collections import deque

from jaal import Jaal
from jaal.datasets import load_got
import pandas as pd

def visualize(ps):

    edge_data = []
    id_modifier = 0
    for p in ps:
        edge_data += [[e["source"] + id_modifier, e["dest"] + id_modifier, str(e["weight"])] for e in p["phenotype"]["edges"]]
        id_modifier += 100
    
    edge_df = pd.DataFrame(edge_data, columns=['from', 'to', 'label'])

    # node_data = [[n["id"]] for n in p["phenotype"]["nodes"]]
    # node_df = pd.DataFrame(node_data, columns=["id"])

    # init Jaal and run server
    Jaal(edge_df).plot(directed=True)


def sigmoid(x):
    return 1 / (1 + (math.e ** x))

WIDTH = 800
HEIGHT = 500
PIXEL_SIZE = 50
JUMP_TIME = 800
OBSTACLE_SPAWN_RATE = 5

NUM_INPUT_NODES = 2
NUM_OUTPUT_NODES = 1
GENERATION_POPULATION_LIMIT = 100

global innovationCounter
innovationCounter = 0
globalEdges = []

global nodeCounter
nodeCounter = NUM_OUTPUT_NODES + NUM_INPUT_NODES + 1

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
        # Default random edges from bias to each output
        addEdge(edges, 0, i, random.uniform(-2, 2))
        i += 1

    return {
        "nodes": nodes,
        "edges": edges
    }

# This is frequently producing a seeming stack overflow
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

    return {
        "jump_timer": 0,
        "fitness": 0,
        "color": "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]),
        "phenotype": {
            "nodes": nodes,
            "edges": new_edges
        }
    }

# Not working
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
    rand = random.randint(0, 100)
    if 0 <= rand <= 40:
        mutate_weight_random(p)
    elif 41 <= rand <= 80:
        mutate_weight_shift(p)
    elif 81 <= rand <= 85:
        mutate_edge(p)
    elif 86 <= rand <= 90:
        mutate_node(p)
    else:
        mutate_enable_disable(p)

# randomly deciding that if two players both have >=85% similar edges than they are same species
def sameSpecies(p1, p2):

    shared_edges = [[e["source"], e["dest"]] for e in p1["phenotype"]["edges"] if [e["source"], e["dest"]] in [[f["source"], f["dest"]] for f in p2["phenotype"]["edges"]]]
    if len(shared_edges) / len(p1["phenotype"]["edges"]) >= 0.85 and len(shared_edges) / len(p2["phenotype"]["edges"]) >= 0.85:
        return True
    else:
        return False


def speciation(players):
    species = []

    for p in players:
        found_species = False
        for s in species:
            if sameSpecies(p, s[0]):
                s.append(p)
                found_species = True
        if not found_species:
            species.append([p])

    return species

def next_generation(players):

    next_gen = []

    # Speciate and eliminate bottom 50% of each species
    species = speciation(players)
    # print("\n")
    for i, s in enumerate(species):
        sorted_s = sorted(s, key=lambda x: x['fitness'], reverse=True)
        # print(f"\nSpecies {i} top fitness: {sorted_s[0]['fitness']}, average fitness: {sum([s['fitness'] for s in sorted_s]) // len(sorted_s)} number of players: {len(sorted_s)}")
        for i in range(len(sorted_s)//2 + 1):
            next_gen.append(sorted_s[i])

    species = speciation(next_gen)

    # Refill to generation limit by randomly selecting players to breed
    while len(next_gen) < GENERATION_POPULATION_LIMIT:
        s = random.choice(species)
        mommy = random.choice(s)
        daddy = random.choice(s)
        baby = cross_over(mommy, daddy)
        if random.randint(0,1) == 1:
            mutate(baby)
        next_gen.append(baby)
        s.append(baby)

    for p in next_gen:
        p["fitness"] = 0

    return next_gen

players = [{
    "jump_timer": 0,
    "fitness": 0,
    "color": "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]),
    "phenotype": s
} for s in [makePlayer() for i in range(GENERATION_POPULATION_LIMIT)]]


# for i in range(90):
#     mutate(players[1])

# players[0]["fitness"] = 1000

# visualize([players[0], players[1], cross_over(players[0], players[1])])

# Game Code


pheight = HEIGHT // PIXEL_SIZE
pwidth = WIDTH // PIXEL_SIZE

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

runs = 0

while True:

    running = True
    score = 0
    counter = 0

    obstacles = []
    runs += 1

    while running:

        screen.fill("#000000")

        score += 1
        livingPlayers = [player for player in players if player["fitness"] == 0]

        if len(livingPlayers) == 0:
            running = False
            break

        # Event handling loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        font = pygame.font.SysFont("Arial", 36)
        score_text = font.render(str(score), True, "#ffffff")
        screen.blit(score_text,(200 - score_text.get_width() // 2, 150 - score_text.get_height() // 2))

        # Add new obstacle
        if random.randint(0, 100) < OBSTACLE_SPAWN_RATE and len([o for o in obstacles if o > WIDTH - JUMP_TIME]) == 0:
            obstacles.append(WIDTH)

        # Move Obstacles
        for i, o in enumerate(obstacles):
            if o < 0:
                obstacles.remove(o)
            else:
                obstacles[i] = o - 5

        # See what each player outputs and draw them
        for p in livingPlayers:

            # subtract from players' jump timers
            if p["jump_timer"] > 0:
                p["jump_timer"] -= 10

            if obstacles:
                input = [1 - min(obstacles)/WIDTH, p["jump_timer"]/JUMP_TIME]
            else:
                input = [0, p["jump_timer"]/JUMP_TIME]
            decision = produce_move(p["phenotype"], input)

            if decision[0] < 0 and p["jump_timer"] == 0:
                p["jump_timer"] = JUMP_TIME

            # Draw player
            if p["jump_timer"] > 500: 
                pygame.draw.rect(screen, p["color"], pygame.Rect(30, HEIGHT - (PIXEL_SIZE * 3), PIXEL_SIZE, PIXEL_SIZE))
            else:
                pygame.draw.rect(screen, p["color"], pygame.Rect(30, HEIGHT - PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE))

        # Draw obstacles
        for o in obstacles:
            pygame.draw.rect(screen, "#00ff00", pygame.Rect(o, HEIGHT - PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE))  

        # Kill neats
        for p in livingPlayers:
            if p["jump_timer"] < 500 and len([o for o in obstacles if o <= 30]) > 0:
                p["fitness"] = score

        # Show screen, and limit FPS to 60 
        if runs % 100 == 0:
            pygame.display.flip()
            clock.tick(60)

    if runs % 10 == 0:
        sorted_players = sorted(players, key=lambda x: x['fitness'], reverse=True)
        print(f"Maximum fitness: {sorted_players[0]['fitness']}, average fitness: {sum([s['fitness'] for s in sorted_players]) // len(sorted_players)}, number of species: {len(speciation(players))}")
        visualize([sorted_players[0]])

    print(f"Run: {runs} complete.")
    players = next_generation(players)
