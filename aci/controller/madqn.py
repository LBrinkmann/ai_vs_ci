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
    return (x+1).unsqueeze(-1).bitwise_and(mask).ne(0).byte()

def onehot(x, m, input_size, device):
    onehot = th.zeros(*x.shape, input_size, dtype=th.float32, device=device)
    x_ = x.clone()
    x_[m] = 0
    onehot.scatter_(-1, x_.unsqueeze(-1), 1)
    onehot[m] = 0
    return onehot


class GRUAgentWrapper(nn.Module):
    def __init__(self, 
            observation_shapes, n_agents, n_actions, net_type, 
            multi_type, device, add_catch=False, global_input_idx=[],
            global_control_idx=[], mix_weights_args=None, **kwargs):
        super(GRUAgentWrapper, self).__init__()
        assert multi_type in ['shared_weights', 'individual_weights']


        self.control_size = len(global_control_idx)
        if observation_shapes[2] is not None:
            secrete_maxval_p1 = observation_shapes[2]['maxval'] + 1
            assert np.log2(secrete_maxval_p1 + 1) == int(np.log2(secrete_maxval_p1 + 1))
            self.control_size += int(np.log2(secrete_maxval_p1 + 1))

        self.n_inputs = observation_shapes[0]['shape'][1]

        self.input_size = n_actions + len(global_input_idx)

        if add_catch:
            self.input_size += 1

        effective_agents = 1 if multi_type == 'shared_weights' else n_agents

        self.models = nn.ModuleList([
            NETS[net_type](
                n_inputs=self.n_inputs, input_size=self.input_size, n_actions=n_actions, device=device, 
                control_size=self.control_size, **kwargs)
            for n in range(effective_agents)
        ])
        self.multi_type = multi_type
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.device = device
        self.global_input_idx = global_input_idx
        self.global_control_idx = global_control_idx
        self.add_catch = add_catch
        self.mix_weights_args = mix_weights_args

        # TODO: this is temporarly
        if mix_weights_args is not None:
            ref_model = self.models[0].state_dict()
            for m in self.models:
                m.load_state_dict(ref_model)

    def mix_weigths(self):
        return self._mix_weigths(**self.mix_weights_args)

    def _mix_weigths(self, noise_factor=0.1, mixing_factor=0.8):
        """
        """
        stacked_state_dict = {}
        state_names = list(self.models[0].state_dict().keys())

        stacked_state_dict = {
            k: th.stack(
                tuple(m.state_dict()[k] for m in self.models),
                dim=0
            ) for k in state_names
        }

        stacked_state_dict = {
            k: v + th.randn_like(v) * v.std() * noise_factor
            for k, v in stacked_state_dict.items()
        }

        stacked_state_dict = {
            k: v * (1-mixing_factor) + v.mean(dim=0) * mixing_factor
            for k, v in stacked_state_dict.items()
        }

        for i in range(len(self.models)):
            self.models[i].load_state_dict({k: v[i] for k, v in stacked_state_dict.items()})


    def forward(self, x, mask, secret=None, globalx=None):
        """
            x: agents, episode, episode_step, neighbors, agent_type
            m: agents, episode, episode_step, neighbors, agent_type
            s: agents, episode, episode_step
            e: 1, episode, episode_step
        """
        x_ci_oh = onehot(x[:,:,:,:,0], mask, self.input_size, self.device) 

        if len(self.global_input_idx) != 0:
            x_ci_oh[:,:,:,:,-len(self.global_input_idx):] = globalx[:,:,:,self.global_input_idx].unsqueeze(-2)

        if self.add_catch:
            catch = (x[:,:,:,:,0] == x[:,:,:,:,1]).type(th.float)
            x_ci_oh[:,:,:,:,-len(self.global_input_idx)-1] = catch
        x_ci_oh[mask] = 0


        if secret is not None:
            control = binary(secret, self.control_size).type(th.float)
            if len(self.global_control_idx) != 0:
                control[:,:,:,-len(self.global_control_idx):] = globalx[:,:,:,self.global_control_idx]
            data = x_ci_oh, mask, control

        elif len(self.global_control_idx) != 0:
            control = globalx[:,:,:,self.global_control_idx]
            data = x_ci_oh, mask, control
        else:
            data = x_ci_oh, mask

        # # random test
        # random_agent = th.randint(high=self.n_agents, size=(1,)).item()
        # random_batch = th.randint(high=x.shape[1], size=(1,)).item()
        # random_step = th.randint(high=x.shape[2], size=(1,)).item()
        # random_neighbor = th.randint(high=(x[random_agent,random_batch,random_step,:,0] != -1).sum(), size=(1,)).item()
        
        # # x_ci_oh test
        # color = x[random_agent,random_batch,random_step,random_neighbor,0]
        # vector = x_ci_oh[random_agent,random_batch,random_step,random_neighbor,:]
        # assert vector[color] == 1, f"{color} {vector}"
        # assert vector[:self.n_actions].sum() == 1

        # if len(self.global_input_idx) != 0:
        #     random_g_idx = th.randint(high=len(self.global_input_idx), size=(1,)).item()
        #     offset = self.input_size - len(self.global_input_idx)
        #     global_vec = globalx[random_agent,random_batch,random_step,self.global_input_idx]
        #     assert vector[offset+random_g_idx] == global_vec[random_g_idx]

        # if self.add_catch:
        #     is_catch = (color == x[random_agent,random_batch,random_step,random_neighbor,1])
        #     idx = - len(self.global_input_idx) - 1
        #     assert vector[idx] == is_catch
        # # end x_ci_oh test

        # # test control
        # if (secret is not None) or (len(self.global_control_idx) != 0):
        #     secret_size = self.control_size - len(self.global_control_idx)
        #     control_vec = control[random_agent,random_batch,random_step]
        # if secret is not None:
        #     binary_secret = np.unpackbits(secret[random_agent, random_batch, random_step].numpy().astype(np.uint8))[-secret_size:][::-1].copy()
        #     binary_secret = th.tensor(binary_secret, dtype=th.float)
        #     assert (control_vec[:secret_size] == binary_secret).all()
        # if len(self.global_control_idx) != 0:
        #     global_vec = globalx[random_agent,random_batch,random_step,self.global_control_idx]
        #     assert (control_vec[-len(self.global_control_idx):] == global_vec).all()
        # # end test control

        if self.multi_type == 'shared_weights':
            data = (
                d.reshape(1, d.shape[0]*d.shape[1], *d.shape[2:])
                for d in data
            )

        q = [
            model(*d)
            for model, *d in zip(self.models, *data)
        ]
        q = th.stack(q)

        if self.multi_type == 'shared_weights':
            q = q.reshape(*x.shape[:3], -1)

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
            gamma, batch_size, target_update_freq, device, mix_freq=None):
        self.agent_type = agent_type
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.device = device
        if model_args['net_type'] != 'gcn':
            self.observation_mode = 'neighbors_mask_secret_envinfo'
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
        self.mix_freq = mix_freq

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
        if (self.mix_freq is not None) and training and (episode % self.mix_freq == 0):
            self.policy_net.mix_weigths()
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