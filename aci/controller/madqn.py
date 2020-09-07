from collections import namedtuple
from functools import reduce
from aci.components.torch_replay_memory import FixedEpisodeMemory
import math
import random
import torch as th
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from aci.neural_modules import NETS



class GRUAgentWrapper(nn.Module):
    def __init__(self, observation_shape, n_agents, n_actions, net_type, multi_type, device, **kwargs):
        super(GRUAgentWrapper, self).__init__()
        assert multi_type in ['shared_weights', 'individual_weights']
        input_shape = observation_shape[0][1]
        effective_agents = 1 if multi_type == 'shared_weights' else n_agents
        self.models = nn.ModuleList([
            NETS[net_type](input_shape=input_shape, n_actions=n_actions, device=device, **kwargs)
            for n in range(effective_agents)
        ])
        self.multi_type = multi_type
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.device = device

    def forward(self, x, m):
        """
            agents x batch x inputs
        """

        if self.multi_type == 'shared_weights':
            o_shape = (*x.shape[:3], self.n_actions)
            x = x.reshape(1, x.shape[0]*x.shape[1], *x.shape[2:])
            m = m.reshape(1, x.shape[0]*x.shape[1], *x.shape[2:])
        onehot = th.zeros(*x.shape, self.n_actions, dtype=th.float32, device=self.device)

        x_ = x.clone()
        x_[m] = 0
        onehot.scatter_(-1, x_.unsqueeze(-1), 1)
        onehot[m] = 0

        q = [
            model(oh, mm)
            for model, oh, mm in zip(self.models, onehot, m)
        ]
        q = th.stack(q)

        if self.multi_type == 'shared_weights':
            q = q.reshape(o_shape)

        return q

    def log(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        for m in self.models:
            m.reset()


class MADQN:
    def __init__(
            self, observation_shapes, n_agents, n_actions, model_args, opt_args, sample_args, agent_type,
            gamma, batch_size, target_update_freq, device):
        self.agent_type = agent_type
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.device = device
        observation_shape = observation_shapes['neighbors_with_mask']
        self.policy_net = GRUAgentWrapper(observation_shape, n_agents, n_actions, device=device, **model_args).to(device)
        self.target_net = GRUAgentWrapper(observation_shape, n_agents, n_actions, device=device, **model_args).to(device)
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), **opt_args)
        self.gamma = gamma
        self.batch_size = batch_size
        self.sample_args = sample_args
        self.target_update_freq = target_update_freq
        self.observation_mode = 'neighbors_with_mask'

    def get_q(self, input, training=None):
        obs, mask = input
        with th.no_grad():
            return self.policy_net(obs.unsqueeze(1).unsqueeze(1), mask.unsqueeze(1).unsqueeze(1)).squeeze(1).squeeze(1)

    def update(self, done, sampler, writer=None, training=None):
        if training and done:
            sample = sampler(agent_type=self.agent_type, batch_size=self.batch_size, mode=self.observation_mode, **self.sample_args)
            if sample is not None:
                self._optimize(*sample)

    def init_episode(self, episode, training):
        self.policy_net.reset()
        if training and (episode % self.target_update_freq == 0):
            self._update_target()

    def _optimize(self, prev_observations, observations, actions, rewards):        
        self.policy_net.reset()
        self.target_net.reset()

        # this will not work with multi yet

        policy_state_action_values = self.policy_net(*prev_observations).gather(-1, actions.unsqueeze(-1))

        next_state_values = th.zeros_like(rewards, device=self.device)
        next_state_values = self.target_net(*observations).max(-1)[0].detach()

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