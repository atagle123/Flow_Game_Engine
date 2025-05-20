from dataclasses import dataclass, field
import numpy as np
import os

@dataclass
class DataStorage:
    states: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    next_states: list = field(default_factory=list)
    dones: list = field(default_factory=list)

    def add(self, obs, next_obs, action, done): 
        self.states.append(obs)
        self.next_states.append(next_obs)
        self.actions.append(action)
        self.dones.append(done)

    def _format_data(self):
        array_states = np.array(self.states).astype(np.float32)
        array_next_states = np.array(self.next_states).astype(np.float32)
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