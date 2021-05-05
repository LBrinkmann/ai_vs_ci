import torch as th


class RandomAgent:
    def __init__(self, observation_shape, n_actions, **_):
        self.n_actions = n_actions

    def get_q(self, actions, **_):
        q = th.rand((*actions.shape[:3], self.n_actions))
        return q
