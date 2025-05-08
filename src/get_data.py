from os import name
import os
from game.maze2d.maze_env import MazeEnv
#import gym
from agent.random_agent import RandomAgent
from data_buffer import DataStorage


def env_loop(env, agent, storage): 
    obs, full_obs = env.reset() # see the format the data is given... 
    done = False
    transitions = None
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
    todo: 
    - produce the game engine
    """
    env = MazeEnv() # TODO register the env
    agent = RandomAgent()
    num_episodes = 1
    storage = DataStorage()
    for _ in range(num_episodes):
        storage = env_loop(env, agent, storage)
    
    filepath = "data"
    filename = "data_maze.npz"
    storage.download_data(filepath, filename)

    
if __name__=="__main__":
    main()
# import gym env
# call agent
# interact with the agent and the env in parallel if desired. 
# store transitions in replay buffer
# save transitions from buffer 