import torch as th


class TitForTat:
    def __init__(self, observation_shape, n_actions):
        self.n_actions = n_actions

    def get_q(self, obs):
        one_hot = th.zeros((obs.shape[0], self.n_actions))
        one_hot = one_hot.scatter_(-1, obs[:,1].unsqueeze(-1), 1)
        return one_hot
