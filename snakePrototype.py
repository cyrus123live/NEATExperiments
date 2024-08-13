import pygame
import random

WIDTH = 500
HEIGHT = 500
PSIZE = 10

pheight = HEIGHT // PSIZE
pwidth = WIDTH // PSIZE

def generate_fruit():
    return [random.randrange(1, pwidth - 1), random.randrange(1, pheight - 1)]

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
running = True

positions = []
for i in range(3):
    positions.append([pwidth - i * PSIZE, pheight // 2])
direction = "right"
fruit = generate_fruit()

while running:

    screen.fill("#000000")
    move = ""

    # Event handling loop
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == 27:
                running = False
            if event.key == 100 and direction != "left":
                direction = "right"
            elif event.key == 115 and direction != "up":
                direction = "down"
            elif event.key == 97 and direction != "right":
                direction = "left"
            elif event.key == 119 and direction != "down":
                direction = "up"

    if direction == "up":
        positions.insert(0, [positions[0][0], (positions[0][1] - 1 + pheight) % pheight])
    elif direction == "down":
        positions.insert(0, [positions[0][0], (positions[0][1] + 1 + pheight) % pheight])
    elif direction == "left":
        positions.insert(0, [(positions[0][0] - 1 + pwidth) % pwidth, positions[0][1]])
    elif direction == "right":
        positions.insert(0, [(positions[0][0] + 1 + pwidth) % pwidth, positions[0][1]])

    if positions[0] != fruit:
        positions.pop(len(positions) - 1)
    else:
        fruit = generate_fruit()

    for i in range(len(positions)):
        pygame.draw.rect(screen, "#ffffff", pygame.Rect(positions[i][0] * PSIZE, positions[i][1] * PSIZE, PSIZE, PSIZE))
    pygame.draw.rect(screen, "#ff0000", pygame.Rect(fruit[0] * PSIZE, fruit[1] * PSIZE, PSIZE, PSIZE))

    pygame.display.flip()
    clock.tick(2)
    