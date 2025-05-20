from src.game.maze2d.maze_env import MazeEnv
from src.agent.random_agent import RandomAgent
from src.utils.data_buffer import DataStorage
import numpy as np

def env_loop(env, agent, storage): 
    game_start_obs = np.zeros((3, 12, 12))
    game_start_action = np.int8(0)
    obs, full_obs = env.reset()
    done = False
    storage.add(game_start_obs, full_obs, game_start_action, done)

    while not done: 
        action = agent.sample_action()
        next_obs, reward, done, next_full_obs = env.step(action)

        storage.add(full_obs, next_full_obs, action, done)
        obs = next_obs
        full_obs = next_full_obs

        #env.render()
    env.close()
    return (storage)


def main():
    """
    """
    filepath = "logs/data"
    filename = "data_maze.npz"
    num_episodes = 100

    env = MazeEnv() # TODO register the env
    agent = RandomAgent()
    storage = DataStorage()

    for _ in range(num_episodes):
        env_loop(env, agent, storage)

    storage.download_data(filepath, filename)

if __name__=="__main__":
    main()