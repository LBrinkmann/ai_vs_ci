from collections import namedtuple
from functools import reduce
from aci.components.torch_replay_memory import FixedEpisodeMemory
import math
import random
import torch as th
import torch.optim as optim
import torch.nn.functional as F
from aci.neural_modules.gru import GRUAgent
import torch.nn as nn
from aci.controller.helper import FloatTransformer, SharedWeightModel, FlatObservation, FakeBatch



class GRUAgentWrapper(nn.Module):
    def __init__(self, observation_shape, n_agents, n_actions, net_type, multi_type, **kwargs):
        super(GRUAgentWrapper, self).__init__()
        effective_agents = 1 if multi_type == 'shared_weights' else n_agents
        input_shape = reduce((lambda x, y: x * y), observation_shape) * n_actions
        self.model = GRUAgent(input_shape=input_shape, n_agents=effective_agents, n_actions=n_actions, **kwargs)
        self.multi_type = multi_type
        self.n_actions = n_actions

    def forward(self, x):
        """
            agents x batch x inputs
        """
        if self.multi_type == 'shared_weights':
            _x = x.reshape(1, x.shape[0]*x.shape[1], *x.shape[2:])
        else:
            _x = x
        __x = th.zeros(*_x.shape, self.n_actions)
        __x.scatter_(-1, _x.unsqueeze(-1), 1)
        __x = __x.reshape(*__x.shape[:3],-1)
        _q = self.model(__x)

        if self.multi_type == 'shared_weights':
            q = _q.reshape(x.shape[0], x.shape[1])
        else:
            q = _q

        return q


    def log(self, *args, **kwargs):
        self.model.log(*args, **kwargs)

    def reset(self, *args, **kwargs):
        self.model.reset(*args, **kwargs)


class MADQN:
    def __init__(
            self, observation_shape, n_agents, n_actions, model_args, opt_args, memory_args, 
            gamma, batch_size, target_update_freq, device):
        self.n_actions = n_actions
        self.device = device
        self.policy_net = GRUAgentWrapper(observation_shape, n_agents, n_actions, **model_args).to(device)
        self.target_net = GRUAgentWrapper(observation_shape, n_agents, n_actions, **model_args).to(device)
        self.memory = FixedEpisodeMemory(
            observation_shape=observation_shape,
            n_agents=n_agents,
            n_actions=n_actions,
            **memory_args)
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), **opt_args)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

    def get_q(self, observations=None):
        with th.no_grad():
            return self.policy_net(observations.unsqueeze(1).unsqueeze(1)).squeeze(1).squeeze(1)

    def update(self, actions, observations, rewards, done, writer=None):
        self.memory.push(observations, actions, rewards)
        if done and (len(self.memory) > self.batch_size):
            self._optimize()

    def init_episode(self, observations, episode, training):
        self.policy_net.reset(1)
        self.memory.init_episode(observations)
        if training and (episode % self.target_update_freq == 0):
            self._update_target()

    def _optimize(self):
        prev_observations, observations, actions, rewards = self.memory.sample(self.batch_size)

        self.policy_net.reset(self.batch_size)
        self.target_net.reset(self.batch_size)

        # this will not work with multi yet

        policy_state_action_values = self.policy_net(prev_observations).gather(-1, actions.unsqueeze(-1))

        next_state_values = th.zeros_like(rewards)
        next_state_values = self.target_net(observations).max(-1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + rewards

        # Compute Huber loss
        loss = F.smooth_l1_loss(policy_state_action_values, expected_state_action_values.unsqueeze(-1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def _update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def log(self, writer):
        self.policy_net.log(writer, prefix='policyNet')