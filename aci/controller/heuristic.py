from collections import namedtuple
import math
import random
import torch as th
import numpy as np
import pandas as pd

from aci.heuristics import HEURISTICS


class HeuristicController:
    def __init__(self, observation_shape, n_agents, n_actions, device, heuristic_name, agent_args):
        self.n_actions = n_actions
        self.device = device
        self.heuristic = HEURISTICS[heuristic_name](observation_shape, n_actions, **agent_args)
        self.last_observations = None
        
    def get_q(self, observation=None):
        if observation is None:
            observation = self.last_observations
        return self.heuristic.get_q(observation)

    def init_episode(self, observations, *_, **__):
        self.last_observations = observations

    def update(self, actions, observations, *_, **__):
        self.last_observations = observations
        pass

    def log(self, writer, details):
        pass
