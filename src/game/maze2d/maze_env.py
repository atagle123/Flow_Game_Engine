import gym
import numpy as np
import pygame
from gym import spaces

class MazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(MazeEnv, self).__init__()
        
        self.maze = np.array([
            [1,0,1,1,1,1,1,1,1,1,1,1],
            [1,0,0,0,0,1,0,0,0,0,0,1],
            [1,0,1,1,0,1,0,1,0,1,0,1],
            [1,0,0,1,0,0,0,1,0,1,0,1],
            [1,1,0,1,1,1,1,1,0,1,0,1],
            [1,0,0,1,0,0,0,0,0,1,0,1],
            [1,0,1,1,0,1,1,1,1,1,0,1],
            [1,0,1,0,0,0,0,0,0,0,0,1],
            [1,0,1,1,1,1,1,1,1,1,0,1],
            [1,0,0,0,0,0,0,0,0,1,0,1],
            [1,1,1,1,1,1,1,1,0,1,0,1],
            [1,1,1,1,1,1,1,1,0,0,0,1]
        ])
        self.height, self.width = self.maze.shape
        self.action_space = spaces.Discrete(4)  # 0=up, 1=right, 2=down, 3=left
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.height, self.width), dtype=np.uint8)

        self.start_pos = [0, 1]
        self.goal_pos = [11, 10]
        self.player_pos = self.start_pos.copy()

        self.window = None
        self.cell_size = 40

    def reset(self):
        self.player_pos = self.start_pos.copy()
        return self._get_obs(), self._get_full_obs()

    def step(self, action):
        x, y = self.player_pos
        moves = [(-1,0), (0,1), (1,0), (0,-1)]  # up, right, down, left
        dx, dy = moves[action]
        nx, ny = x + dx, y + dy

        if 0 <= nx < self.height and 0 <= ny < self.width and self.maze[nx][ny] == 0:
            self.player_pos = [nx, ny]

        done = self.player_pos == self.goal_pos
        reward = 1 if done else -0.01

        return self._get_obs(), reward, done, self._get_full_obs() # 1 channel is the maze, 2 is the player pos, 3 the goal pos

    def _get_full_obs(self):
        pos_player_grid = np.zeros_like(self.maze)
        pos_player_grid[self.player_pos[0]][self.player_pos[1]] = 1

        goal_pos_grid = np.zeros_like(self.maze)
        goal_pos_grid[self.goal_pos[0]][self.goal_pos[1]] = 1

        full_obs = np.stack([self.maze, pos_player_grid, goal_pos_grid], axis=0)
        return(full_obs)


    def _get_obs(self):
        obs = np.array(self.player_pos)
        return obs

    def render(self, mode="human"):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.width*self.cell_size, self.height*self.cell_size))
            pygame.display.set_caption("Maze Environment")

        self.window.fill((0, 0, 0))
        for y in range(self.height):
            for x in range(self.width):
                color = (255, 255, 255) if self.maze[y][x] == 1 else (30, 30, 30)
                rect = pygame.Rect(x*self.cell_size, y*self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.window, color, rect)

        # Draw player
        px, py = self.player_pos[1], self.player_pos[0]
        pygame.draw.circle(self.window, (255, 0, 0),
                           (px*self.cell_size + self.cell_size//2, py*self.cell_size + self.cell_size//2),
                           self.cell_size//3)
        pygame.display.flip()

    def close(self):
        if self.window:
            pygame.quit()
            self.window = None
