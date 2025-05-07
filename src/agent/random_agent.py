import numpy as np

class RandomAgent:
    def __init__(self, action_space= None, action_type = "discrete"):
        self.action_type = action_type
       # if action_type == "discrete":
       #    self.sampling_fn = self.discrete_sample
            #self.action_space = action_space
            # TODO implment; this has to take the action space to allow sampling from continuous and discrete action spaces.

    #def continuous_sample(self):
        # TODO implement
        #np.random.uniform(low, high, size)

    #def discrete_sample(self): 

    def sample_action(self):
        sample = np.random.choice([0, 1, 2, 3])

        return (sample)