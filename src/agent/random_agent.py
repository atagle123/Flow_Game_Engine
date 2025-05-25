import numpy as np

class RandomAgent:
    def __init__(self):
        pass
    def sample_action(self):
        sample = np.random.choice([0, 1, 2, 3]).astype(np.int8)

        return (sample)