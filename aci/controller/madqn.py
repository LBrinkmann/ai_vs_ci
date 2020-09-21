from collections import namedtuple
from functools import reduce
import math
import random
import torch as th
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from aci.neural_modules import NETS




def binary(x, bits):
    mask = 2**th.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


class GRUAgentWrapper(nn.Module):
    def __init__(self, 
            observation_shapes, n_agents, n_actions, net_type, 
            multi_type, device, **kwargs):
        super(GRUAgentWrapper, self).__init__()
        assert multi_type in ['shared_weights', 'individual_weights']

        if observation_shapes[2] is None:
            self.secrete_size = 0
        else:
            secrete_maxval = observation_shapes[2]['maxval']
            assert np.log2(secrete_maxval + 1) == int(np.log2(secrete_maxval + 1))
            self.secrete_size = int(np.log2(secrete_maxval + 1))

        n_inputs = observation_shapes[0]['shape'][1]
        self.input_size = n_actions + 1

        effective_agents = 1 if multi_type == 'shared_weights' else n_agents

        self.models = nn.ModuleList([
            NETS[net_type](
                n_inputs=n_inputs, input_size=self.input_size, n_actions=n_actions, device=device, 
                secrete_size=self.secrete_size, **kwargs)
            for n in range(effective_agents)
        ])
        self.multi_type = multi_type
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.device = device

    def forward(self, x, m, s=None, e=None):
        """
            agents x batch x inputs
        """

        if self.multi_type == 'shared_weights':
            o_shape = (*x.shape[:3], self.n_actions)
            x = x.reshape(1, x.shape[0]*x.shape[1], *x.shape[2:])
            m = m.reshape(1, x.shape[0]*x.shape[1], *x.shape[2:])
            if e is not None:
                e = e.unsqueeze(-1).repeat(self.n_agents,1,1,1)
                e = e.reshape(1, e.shape[0]*e.shape[1], *e.shape[2:])
            if s is not None:
                s = s.reshape(1, s.shape[0]*s.shape[1], *s.shape[2:])
        else:
            if e is not None:
                e = e.unsqueeze(-1)

        onehot = th.zeros(*x.shape, self.input_size, dtype=th.float32, device=self.device)
        x_ = x.clone()
        x_[m] = 0
        onehot.scatter_(-1, x_.unsqueeze(-1), 1)
        if e is not None:
            onehot[:,:,:,-1] = e
        onehot[m] = 0

        if s is not None:
            binary_secrets = binary(s, self.secrete_size).type(th.float)
            data = onehot, m, binary_secrets
        else:
            data = onehot, m

        q = [
            model(*d)
            for model, *d in zip(self.models, *data)
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


class OneHotTransformer(nn.Module):
    def __init__(self, batch_size, observation_shape, n_agents, n_actions, model, device):
        super(OneHotTransformer, self).__init__()
        self.model = model
        self.device = device
        self.n_actions = n_actions

    def forward(self, x, links, mask):
        n_agents, batch_size, seq_length = x.shape

        shift_end = seq_length * batch_size * n_agents

        shift = th.arange(start=0, end=shift_end, step=n_agents, device=self.device)
        shift = shift.reshape(batch_size, seq_length).unsqueeze(0).repeat(n_agents, 1, 1)

        edge_index = links + shift.unsqueeze(-1).unsqueeze(-1)
        edge_index = edge_index[~mask].permute(1,0)
        x = x.reshape(-1)

        onehot = th.zeros(len(x), self.n_actions, dtype=th.float32, device=self.device)
        x_ = x.clone()
        onehot.scatter_(-1, x_.unsqueeze(-1), 1)
        y = self.model(onehot, edge_index)
        y = y.reshape(n_agents, batch_size, seq_length, -1)
        return y
    
    def log(self, *args, **kwargs):
        self.model.log(*args, **kwargs)

    def reset(self):
        if getattr(self.model, "reset", False):
            self.model.reset()




class MADQN:
    def __init__(
            self, observation_shapes, n_agents, n_actions, model_args, opt_args, sample_args, agent_type,
            gamma, batch_size, target_update_freq, device):
        self.agent_type = agent_type
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.device = device
        if model_args['net_type'] != 'gcn':
            self.observation_mode = 'neighbors_mask_secret_envstate'
            observation_shapes = observation_shapes[self.observation_mode]
            self.policy_net = GRUAgentWrapper(observation_shapes, n_agents, n_actions, device=device, **model_args).to(device)
            self.target_net = GRUAgentWrapper(observation_shapes, n_agents, n_actions, device=device, **model_args).to(device)
        else:
            self.observation_mode = 'matrix'
            model_args.pop('net_type')
            target_model = NETS['gcn'](n_actions=n_actions, device=device, **model_args).to(device)
            policy_model = NETS['gcn'](n_actions=n_actions, device=device, **model_args).to(device)
            self.policy_net = OneHotTransformer(
                model=target_model, batch_size=batch_size, observation_shape=(1,), n_agents=n_agents, n_actions=n_actions, device=device)
            self.target_net = OneHotTransformer(
                model=policy_model, batch_size=batch_size, observation_shape=(1,), n_agents=n_agents, n_actions=n_actions, device=device)
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), **opt_args)
        self.gamma = gamma
        self.batch_size = batch_size
        self.sample_args = sample_args
        self.target_update_freq = target_update_freq

    def get_q(self, input, training=None):
        with th.no_grad():
            return self.policy_net(*(i.unsqueeze(1).unsqueeze(1) if i is not None else None for i in input)).squeeze(1).squeeze(1)

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