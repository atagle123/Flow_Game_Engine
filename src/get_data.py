from game.maze2d.maze_env import MazeEnv
#import gym
from agent.random_agent import RandomAgent
from buffer import ReplayBuffer


def env_loop(env, agent): 
    obs = env.reset() # see the format the data is given... 
    done = False

    while not done: 
        action = agent.sample_action()
        obs, reward, done, _, _ = env.step(action)

        # store transitions... 
    return (transitions)


def main():
    """
    todo: 
    - init buffer, env, agent
    - make various cycles of interactions with the env (function episode)
    - store the episodes in the replay buffer
    - format the data to the desired form, and store it
    - train the flow matching model with data
    - produce the game engine
    """
    buffer = ReplayBuffer()
    env = MazeEnv # TODO register the env
    agent = RandomAgent()

    

# import gym env
# call agent
# interact with the agent and the env in parallel if desired. 
# store transitions in replay buffer
# save transitions from buffer 