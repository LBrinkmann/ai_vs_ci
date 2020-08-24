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



class GRUAgentWrapper(nn.Module):
    def __init__(self, observation_shape, n_agents, n_actions, net_type, multi_type, device, **kwargs):
        super(GRUAgentWrapper, self).__init__()
        assert multi_type in ['shared_weights', 'individual_weights']

        effective_agents = 1 if multi_type == 'shared_weights' else n_agents
        input_shape = reduce((lambda x, y: x * y), observation_shape) * n_actions
        self.models = nn.ModuleList([
            GRUAgent(input_shape=input_shape, n_actions=n_actions, device=device, **kwargs)
            for n in range(effective_agents)
        ])
        self.multi_type = multi_type
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.device = device

    def forward(self, x):
        """
            agents x batch x inputs
        """
        if self.multi_type == 'shared_weights':
            o_shape = (*x.shape[:3], self.n_actions)
            x = x.reshape(1, x.shape[0]*x.shape[1], *x.shape[2:])
        onehot = th.zeros(*x.shape, self.n_actions, device=self.device)
        onehot.scatter_(-1, x.unsqueeze(-1), 1)
        onehot = onehot.reshape(*onehot.shape[:3],-1)
        q = [
            m(oh)
            for m, oh in zip(self.models, onehot)
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
            self, observation_shape, n_agents, n_actions, model_args, opt_args, memory_args, 
            gamma, batch_size, target_update_freq, device):
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.observation_shape = observation_shape
        self.device = device
        self.policy_net = GRUAgentWrapper(observation_shape, n_agents, n_actions, device=device, **model_args).to(device)
        self.target_net = GRUAgentWrapper(observation_shape, n_agents, n_actions, device=device, **model_args).to(device)
        self.memory = FixedEpisodeMemory(
            observation_shape=observation_shape,
            n_agents=n_agents,
            n_actions=n_actions,
            device=device,
            **memory_args)
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), **opt_args)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

    def get_q(self, observations=None, training=None):
        with th.no_grad():
            return self.policy_net(observations.unsqueeze(1).unsqueeze(1)).squeeze(1).squeeze(1)

    def update(self, actions, observations, rewards, done, writer=None, training=None):
        if training:
            self.memory.push(observations, actions, rewards)
            if done and (len(self.memory) > self.batch_size):
                self._optimize()

    def init_episode(self, observations, episode, training):
        self.policy_net.reset()
        if training:
            self.memory.init_episode(observations)
        if training and (episode % self.target_update_freq == 0):
            self._update_target()

    def _optimize(self):
        prev_observations, observations, actions, rewards = self.memory.sample(self.batch_size)

        # prev_observations: agents, batch_size, episode_steps, neighbors + 1
        # actions: agents, batch_size, episode_steps
        # rewards: agents, batch_size, episode_steps

        # assert prev_observations.shape[:2] == (self.n_agents, self.batch_size)
        # assert rewards.shape[:3] == actions.shape[:3] == prev_observations.shape[:3] == observations.shape[:3]
        # assert (prev_observations[:,:,1:] == observations[:,:,:-1]).all()
        # # for ci only
        # assert (observations[:,:,:,0] == actions).all()
        
        self.policy_net.reset()
        self.target_net.reset()

        # this will not work with multi yet

        policy_state_action_values = self.policy_net(prev_observations).gather(-1, actions.unsqueeze(-1))

        next_state_values = th.zeros_like(rewards, device=self.device)
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