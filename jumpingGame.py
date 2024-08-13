import pygame
import random

WIDTH = 800
HEIGHT = 500
PIXEL_SIZE = 50
JUMP_TIME = 800
OBSTACLE_SPAWN_RATE = 5

pheight = HEIGHT // PIXEL_SIZE
pwidth = WIDTH // PIXEL_SIZE

score = 0
counter = 0

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
running = True

obstacles = []
jump_timer = 0

while running:

    screen.fill("#000000")

    score += 1

    font = pygame.font.SysFont("Arial", 36)
    score_text = font.render(str(score), True, "#ffffff")
    screen.blit(score_text,(200 - score_text.get_width() // 2, 150 - score_text.get_height() // 2))

    # Add new obstacle
    if random.randint(0, 100) < OBSTACLE_SPAWN_RATE and len([o for o in obstacles if o > WIDTH - JUMP_TIME]) == 0:
        obstacles.append(WIDTH)

    # Process Obstacles
    for i, o in enumerate(obstacles):
        if o < 0:
            obstacles.remove(o)
        else:
            obstacles[i] = o - 5

    if jump_timer > 0:
        jump_timer -= 10

    # Process events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == 27:
                quit()
            if event.key == 32 and jump_timer == 0:
                jump_timer = JUMP_TIME

    # Draw everything
    if jump_timer > 500: 
        pygame.draw.rect(screen, "#ff0000", pygame.Rect(30, HEIGHT - (PIXEL_SIZE * 3), PIXEL_SIZE, PIXEL_SIZE))
    else:
        pygame.draw.rect(screen, "#ff0000", pygame.Rect(30, HEIGHT - PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE))
    for o in obstacles:
        pygame.draw.rect(screen, "#00ff00", pygame.Rect(o, HEIGHT - PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE))  

    # Kill player
    if jump_timer == 0 and len([o for o in obstacles if o <= 30]) > 0:
        score = 0
        obstacles = []

    # Show screen, and limit FPS to 60 
    pygame.display.flip()
    clock.tick(60)