from typing import List
from dataclasses import dataclass, field
import numpy as np
import os

@dataclass
class DataStorage:
    states: List = field(default_factory=list)
    actions: List = field(default_factory=list)
    next_states: List = field(default_factory=list)
    dones: List = field(default_factory=list)

    def add(self, obs, next_obs, action, done):
        self.states.append(obs)
        self.next_states.append(next_obs)
        self.actions.append(action)
        self.dones.append(done)

    def _format_data(self):
        array_states = np.array(self.states)
        array_next_states = np.array(self.next_states)
        array_actions = np.array(self.actions)
        array_dones = np.array(self.dones)

        return {
            "states": array_states,
            "actions": array_actions,
            "next_states": array_next_states,
            "dones": array_dones
        }

    def download_data(self, filepath, filename):
        data = self._format_data()
        
        os.makedirs(filepath, exist_ok=True)
        np.savez(f"{filepath}/{filename}", **data)








# replay buffer storing all of the transitions (action, state, next_state)

# visualize data class that takes the maze, goal position and transitions to deliver a matrix of shape (MH, MW, T),
#once that take a random agent get data from the maze and store it 

# the flow model recieves data x_t -> x_t+1 (MH, MW), (MH,MW)
# let use that x_0 its always the same (the starting point)

# agent interacting with the env 