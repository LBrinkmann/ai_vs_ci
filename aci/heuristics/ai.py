import torch as th


class HeuristicAI1:
    def __init__(self, observation_shape, n_actions):
        self.n_actions = n_actions

    def get_q(self, obs):

        state = obs[:,0]

        one_hot = th.zeros(state.shape + (self.n_actions,))
        one_hot = one_hot.scatter_(-1, state.unsqueeze(-1), 1)

        return one_hot
