from collections import namedtuple
from aci.components.torch_replay_memory import ReplayMemory
import math
import random
import torch as th
import torch.optim as optim
import torch.nn.functional as F
from aci.neural_modules.linear_q_function import LinearFunction
from aci.neural_modules.rnn import RNN
from aci.neural_modules.medium_conv import MediumConv
from aci.neural_modules.learning_heuristic import LearningHeuristic
import torch.nn as nn

from aci.components.simple_observation_cache import SimpleCache
from aci.utils.array_to_df import using_multiindex, map_columns, to_alphabete


# def pad_circular(x, pad):
#     """

#     :param x: shape [H, W]
#     :param pad: int >= 0
#     :return:
#     """
#     x = torch.cat([x, x[0:pad]], dim=0)
#     x = torch.cat([x, x[:, 0:pad]], dim=1)
#     x = torch.cat([x[-2 * pad:-pad], x], dim=0)
#     x = torch.cat([x[:, -2 * pad:-pad], x], dim=1)

#     return x



class OneHotTransformer(nn.Module):
    def __init__(self, batch_size, observation_shape, n_agents, n_actions, model):
        super(OneHotTransformer, self).__init__()
        self.model = model
        self.onehot = th.FloatTensor(batch_size, n_agents, *observation_shape, n_actions)

    def forward(self, x):
           # In your for loop
        self.onehot.zero_()
        self.onehot.scatter_(-1, x.unsqueeze(-1), 1)
        y = self.model(self.onehot)
        return y
    
    def log(self, *args, **kwargs):
        self.model.log(*args, **kwargs)


class FloatTransformer(nn.Module):
    def __init__(self, model):
        super(FloatTransformer, self).__init__()
        self.model = model

    def forward(self, x):
        _x = x.type(th.float32)
        y = self.model(_x)
        return y
    
    def log(self, *args, **kwargs):
        self.model.log(*args, **kwargs)

    def reset(self):
        if getattr(self.model, "reset", False):
            self.model.reset()


class FlatHistory(nn.Module):
    def __init__(self, model):
        super(FlatHistory, self).__init__()
        self.model = model

    def forward(self, x):
        _x = x.reshape(x.shape[0],*x.shape[2:])
        y = self.model(_x)
        return y
    
    def log(self, *args, **kwargs):
        self.model.log(*args, **kwargs)

    def reset(self):
        if getattr(self.model, "reset", False):
            self.model.reset()


class SharedWeightModel(nn.Module):
    def __init__(self, model):
        super(SharedWeightModel, self).__init__()
        self.model = model

    def forward(self, x):
        _x = x.unsqueeze(1)
        _y = self.model(_x)
        y = _y.squeeze(1)
        return y

    def reset(self):
        if getattr(self.model, "reset", False):
            self.model.reset()
    
    def log(self, *args, **kwargs):
        self.model.log(*args, **kwargs)


class FakeBatch(nn.Module):
    def __init__(self, model):
        super(FakeBatch, self).__init__()
        self.model = model

    def forward(self, x):
        _x = x.unsqueeze(0)
        _y = self.model(_x)
        y = _y.squeeze(0)
        return y

    def reset(self):
        if getattr(self.model, "reset", False):
            self.model.reset()
    
    def log(self, *args, **kwargs):
        self.model.log(*args, **kwargs)


def create_model(cache_size, observation_shape, n_agents, n_actions, net_type, multi_type, **kwargs):
    effective_agents = 1 if multi_type == 'shared_weights' else n_agents
    batch_size = n_agents if multi_type == 'shared_weights' else 1
    if net_type == 'linear':
        model = LinearFunction(observation_shape=(cache_size,*observation_shape), n_agents=effective_agents, n_actions=n_actions)
        # model = FlatObservation(model, observation_shape=(cache_size,*observation_shape),)
        model = OneHotTransformer(
            observation_shape=(cache_size,*observation_shape), n_actions=n_actions, n_agents=effective_agents, 
            model=model, batch_size=batch_size)
    elif net_type == 'rnn':
        model = RNN(batch_size=batch_size, observation_shape=(cache_size,*observation_shape), n_agents=effective_agents, n_actions=n_actions, **kwargs)
        model = OneHotTransformer(
            observation_shape=(cache_size,*observation_shape), n_actions=n_actions, n_agents=effective_agents, 
            model=model, batch_size=batch_size)


    if multi_type == 'shared_weights':
        model = SharedWeightModel(model)
    elif multi_type == 'individual_weights':
        model = FakeBatch(model)
    else:
        raise NotImplementedError(f"Multi type not found: {multi_type}")

    return model


class MAQ:
    def __init__(
            self, observation_shape, n_agents, n_actions, gamma, cache_size, model_args, opt_args, device):
        self.n_agents = n_agents
        self.gamma = gamma
        self.cache_size = cache_size
        self.q_func = create_model(
            cache_size, observation_shape, n_agents, n_actions, **model_args).to(device)
        self.optimizer = optim.RMSprop(self.q_func.parameters(), **opt_args)
        # th.autograd.set_detect_anomaly(True)

    def get_q(self, observations=None):
        """ Retrieves q values for all possible actions. 

        If observations are given, they are used. Otherwise last observations are used.

        """
        if observations is not None:
            historized_obs = self.cache.add_get(observations)
        else:
            historized_obs = self.cache.get()
        self.q_func.eval()
        with th.no_grad():
            return self.q_func(historized_obs)

    def init_episode(self, observations):
        self.cache = SimpleCache(observations, self.cache_size)
        self.q_func.reset()

    def update(self, actions, observations, rewards, done, writer=None):
        prev_historized_obs = self.cache.get()
        historized_obs = self.cache.add_get(observations)

        with th.no_grad():
            next_state_values = self.q_func(historized_obs).max(-1)[0]
            expected_state_action_values = (next_state_values * self.gamma) + rewards

        policy_state_action_values = self.q_func(prev_historized_obs).gather(-1, actions.unsqueeze(-1))

        # Compute Huber loss
        loss = F.smooth_l1_loss(policy_state_action_values, expected_state_action_values.unsqueeze(-1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.q_func.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if writer and done:
            self.log(writer)

    def log(self, writer):
        if writer.check_on(on='weights'):
            self.q_func.log(writer, prefix='q_func')


