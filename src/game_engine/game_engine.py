import pygame


class GameEngine: 
    def __init__(self, model):
        self.model = model

    def format_observations(self, obs):
        # convert it into a maze 2x2
        # find the player pos...
        # find the goal... 
        # find the walls and blank spaces...
        return (obs)

    def play(self):
        CELL_SIZE = 40
        GRID_WIDTH, GRID_HEIGHT = 12, 12
        WIDTH, HEIGHT = CELL_SIZE * GRID_WIDTH, CELL_SIZE * GRID_HEIGHT
        WHITE, BLACK, RED, GREEN = (255,255,255), (0,0,0), (255,0,0), (0,255,0)
        # === Game Setup ===
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Maze Game")
        clock = pygame.time.Clock()

        #player_pos = [1, 0]
        #goal_pos = [GRID_WIDTH - 2, GRID_HEIGHT - 1]
        KEY_TO_ACTION = {
                        pygame.K_UP: 0,
                        pygame.K_DOWN: 1,
                        pygame.K_LEFT: 2,
                        pygame.K_RIGHT: 3,
                    }

        # === Game Loop ===
        running = True
        # visualize one time first the init... 
        # obs = ... 
        while running:
            clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Movement
            keys = pygame.key.get_pressed()

            action = None
            for key, act in KEY_TO_ACTION.items():
                if keys[key]:
                    action = act
                    break

            
            next_obs = self.model(obs, action) # maybe sample a batch and avg the results to a more robust result. 
              #  running = False
            # format obs... 
            draw(next_obs)
            obs = next_obs

        pygame.quit()
        sys.exit()



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