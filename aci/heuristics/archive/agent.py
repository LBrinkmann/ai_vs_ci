import torch as th
import torch.nn.functional as tf

class HeuristicAgent1:
    def __init__(self, observation_shape, n_actions, self_weight):
        self.n_actions = n_actions
        self.self_weight = self_weight

    def get_q(self, x):
        one_hot = th.zeros(x.shape + (self.n_actions,))
        one_hot = one_hot.scatter_(-1, x.unsqueeze(-1), 1)

        other = one_hot[:, 1:].sum(-2)
        this = one_hot[:, 0]

        q = this * self.self_weight - other
        return q
