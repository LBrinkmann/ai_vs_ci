from collections import namedtuple
import math
import random
import torch
import numpy as np


class SimpleGraphAgent:
    def __init__(self, observation_shape, n_actions, device, **_):
        self.n_actions = n_actions
        self.device = device
        
    def get_action(self, observation):
        observation = observation.cpu().numpy()
        own_state = observation[0]
        other_state = observation[1:]
        unique_elements, counts_elements = np.unique(other_state, return_counts=True)
        default_dict = {i: 0 for i in range(self.n_actions)}
        count_dict = {i: c for i, c in zip(unique_elements, counts_elements)}
        count_dict = {**default_dict, **count_dict}
        is_min = count_dict.values() == count_dict.values().min()
        return torch.tensor(np.random.choice(count_dict.keys(), size=1, p=is_min/is_min.sum()))

    def optimize(self):
        pass

    def update(self):
        pass

    def push_transition(self, *transition):
        pass

    def log(self, writer, step, details):
        pass