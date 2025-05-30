import pygame
import numpy as np
import sys
import jax.numpy as jnp

CELL_SIZE = 40
GRID_WIDTH, GRID_HEIGHT = 12, 12
WIDTH, HEIGHT = CELL_SIZE * GRID_WIDTH, CELL_SIZE * GRID_HEIGHT
WHITE, BLACK, RED, GREEN = (255, 255, 255), (0, 0, 0), (255, 0, 0), (0, 255, 0)


class GameEngine:
    def __init__(self, model):
        self.model = model

    def play(self, n_flow_steps: int = 10):
        # === Game Setup ===
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Maze Game")
        clock = pygame.time.Clock()

        KEY_TO_ACTION = {  # 0=up, 1=right, 2=down, 3=left
            pygame.K_UP: 0,
            pygame.K_DOWN: 2,
            pygame.K_LEFT: 3,
            pygame.K_RIGHT: 1,
        }

        # === Game Loop ===
        running = True
        game_start_action = np.array([0])

        maze = np.array(
            [
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
        )

        player_pos = np.zeros((12, 12), dtype=float)
        player_pos[0, 1] = 1.0
        goal_pos = np.zeros((12, 12), dtype=float)
        goal_pos[-1, -2] = 1.0
        init_obs = np.stack((maze, player_pos, goal_pos), axis=0)
        init_obs = np.expand_dims(init_obs, axis=0)

        self.draw(init_obs)
        obs = self.model.sample(
            init_obs, game_start_action, n_steps=n_flow_steps
        )  # init obs to start the game

        self.draw(obs)
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

            if action is not None:
                sampling_attempts = 2
                while sampling_attempts > 0:
                    try:
                        obs = self.model.sample(
                            obs, np.array([action]), n_steps=n_flow_steps
                        )
                        self.draw(obs)
                        break
                    except Exception as e:
                        print(f"Sampling failed: {e}. Retrying...")
                        sampling_attempts -= 1

        pygame.quit()
        sys.exit()

    def _format_observations(self, obs):
        obs = jnp.round(jnp.squeeze(obs))
        maze = obs[0, :, :]
        player = obs[1, :, :]
        goal = obs[2, :, :]

        player_pos = jnp.argwhere(player == 1)
        goal_pos = jnp.argwhere(goal == 1)
        return (maze, player_pos[0], goal_pos[0])

    def draw(self, obs):
        maze, player_pos, goal_pos = self._format_observations(obs)
        self.screen.fill(BLACK)
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if maze[y, x] == 1:
                    pygame.draw.rect(self.screen, WHITE, rect)
        # Player
        py, px = player_pos
        pygame.draw.rect(
            self.screen,
            RED,
            (px * CELL_SIZE + 5, py * CELL_SIZE + 5, CELL_SIZE - 10, CELL_SIZE - 10),
        )
        # Goal
        gy, gx = goal_pos
        pygame.draw.rect(
            self.screen,
            GREEN,
            (gx * CELL_SIZE + 5, gy * CELL_SIZE + 5, CELL_SIZE - 10, CELL_SIZE - 10),
        )
        pygame.display.flip()
