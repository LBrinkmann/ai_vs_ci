import torch as th


class RandomAgent:
    def __init__(self, observation_shape, n_actions, device, **_):
        self.n_actions = n_actions
        self.device = device

    def get_q(self, actions, **_):
        q = th.rand((*actions.shape[:3], self.n_actions), device=self.device)
        return q
