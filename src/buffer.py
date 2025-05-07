from dataclasses import dataclass, field
import numpy as np

@dataclass
class ReplayBuffer:
    #episodes: list = field(init=False)
    states: np.ndarray = field(init=False)
    actions: np.ndarray = field(init=False)
    rewards: np.ndarray = field(init=False)
    next_states: np.ndarray = field(init=False)
    dones: np.ndarray = field(init=False)

    def __init__(self):
        self.states = np.zeros((self.capacity, *self.state_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, *self.state_shape), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
    
    def add(self, obs, action, next_obs, rewards, dones):
        "implement"

    def get (self):


# replay buffer storing all of the transitions (action, state, next_state)

# visualize data class that takes the maze, goal position and transitions to deliver a matrix of shape (MH, MW, T),
#once that take a random agent get data from the maze and store it 

# the flow model recieves data x_t -> x_t+1 (MH, MW), (MH,MW)
# let use that x_0 its always the same (the starting point)

# agent interacting with the env 