from game.maze2d.maze_env import MazeEnv
#import gym
from agent.random_agent import RandomAgent
from data_buffer import DataStorage


def env_loop(env, agent, storage): 
    obs, full_obs = env.reset() 
    done = False
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
        storage = env_loop(env, agent, storage)

    storage.download_data(filepath, filename)

if __name__=="__main__":
    main()