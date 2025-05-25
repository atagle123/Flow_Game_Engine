import pygame
import random
import sys

# === Config ===
CELL_SIZE = 40
GRID_WIDTH, GRID_HEIGHT = 12, 12
WIDTH, HEIGHT = CELL_SIZE * GRID_WIDTH, CELL_SIZE * GRID_HEIGHT
WHITE, BLACK, RED, GREEN = (255,255,255), (0,0,0), (255,0,0), (0,255,0)

# === Game Setup ===
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Maze Game")
clock = pygame.time.Clock()

#maze = generate_maze(GRID_WIDTH, GRID_HEIGHT)
MAZE = [
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
    [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
]
player_pos = [1, 0]
goal_pos = [GRID_WIDTH - 2, GRID_HEIGHT - 1]

def draw():
    screen.fill(BLACK)
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            rect = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if MAZE[y][x] == 1:
                pygame.draw.rect(screen, WHITE, rect)
    # Player
    px, py = player_pos
    pygame.draw.rect(screen, RED, (px*CELL_SIZE+5, py*CELL_SIZE+5, CELL_SIZE-10, CELL_SIZE-10))
    # Goal
    gx, gy = goal_pos
    pygame.draw.rect(screen, GREEN, (gx*CELL_SIZE+5, gy*CELL_SIZE+5, CELL_SIZE-10, CELL_SIZE-10))
    pygame.display.flip()

# === Game Loop ===
running = True
while running:
    clock.tick(30)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Movement
    keys = pygame.key.get_pressed()
    dx, dy = 0, 0
    if keys[pygame.K_LEFT]: dx = -1
    if keys[pygame.K_RIGHT]: dx = 1
    if keys[pygame.K_UP]: dy = -1
    if keys[pygame.K_DOWN]: dy = 1

    nx, ny = player_pos[0] + dx, player_pos[1] + dy
    if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and MAZE[ny][nx] == 0:
        player_pos = [nx, ny]

    if player_pos == goal_pos:
        print("🎉 You reached the goal!")
        pygame.time.delay(1000)
        running = False

    draw()

pygame.quit()
sys.exit()