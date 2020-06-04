from collections import namedtuple
import math
import random
import torch
import numpy as np
import pandas as pd


class SimpleGraphAgent:
    def __init__(self, observation_shape, n_actions, device, **_):
        self.n_actions = n_actions
        self.device = device
        
    def get_action(self, observation):
        diff = (observation[:, 1:] - observation[:, [0]]).abs().sum(1)
        p_change = 0.5 * diff

        s_change = torch.bernoulli(p_change)
        action = s_change * (1 - observation[:, 0]) + (1-s_change) * observation[:, 0]
        return action.type(torch.int64)

    def optimize(self):
        pass

    def update(self):
        pass

    def push_transition(self, *transition):
        pass

    def log(self, writer, step, details):
        pass