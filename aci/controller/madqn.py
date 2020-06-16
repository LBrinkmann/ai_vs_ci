from collections import namedtuple
from aci.components.torch_replay_memory import ReplayMemory
import math
import random
import torch as th
import torch.optim as optim
import torch.nn.functional as F
from aci.neural_modules.logistic_regression import LogisticRegression
from aci.neural_modules.medium_conv import MediumConv
from aci.neural_modules.learning_heuristic import LearningHeuristic
from aci.neural_modules.simple_rnn import RecurrentModel
import torch.nn as nn


class FloatTransformer(nn.Module):
    def __init__(self, model):
        super(FloatTransformer, self).__init__()
        self.model = model

    def forward(self, x, h):
        _x = x.type(th.float32)
        y, h = self.model(_x, h)
        return y, h

    def init_hidden(self, batch_size):
        return self.model.init_hidden(batch_size)

    def log(self, *args, **kwargs):
        self.model.log(*args, **kwargs)



class SharedWeightModel(nn.Module):
    def __init__(self, model, n_agents):
        super(SharedWeightModel, self).__init__()
        self.model = model
        self.n_agents = n_agents

    def forward(self, x, h):
        seq_length = x.shape[0]
        batch_size = x.shape[1]
        n_agents = x.shape[2]
        _x = x.reshape(seq_length, -1, *x.shape[3:])
        y, h_ = self.model(_x, h)
        y = y.reshape(seq_length, batch_size, n_agents, *y.shape[2:])
        return y, h_

    def init_hidden(self, batch_size):
        return self.model.init_hidden(batch_size * self.n_agents)
    
    def log(self, *args, **kwargs):
        self.model.log(*args, **kwargs)


def create_net(observation_shape, n_agents, n_actions, net_type, multi_type, **kwargs):
    if net_type == 'medium_conv':
        model = MediumConv(*observation_shape, n_actions, **kwargs)
        model = FloatTransformer(model)
    elif net_type == 'logistic':
        model = LogisticRegression(*observation_shape, n_actions, **kwargs)
        model = FloatTransformer(model)
    elif net_type == 'learningheuristic':
        model = LearningHeuristic(*observation_shape, n_actions, **kwargs)
    elif net_type == 'rnn':
        model = RecurrentModel(*observation_shape, n_actions, **kwargs)
        model = FloatTransformer(model)
    else:
        raise NotImplementedError(f"Net type not found: {net_type}")
    
    if multi_type == 'shared_weights':
        model = SharedWeightModel(model, n_agents)
    else:
        raise NotImplementedError(f"Multi type not found: {multi_type}")

    return model


class MADQN:
    def __init__(
            self, observation_shape, n_agents, n_actions, policy_args, opt_args, gamma, device):
        self.n_actions = n_actions
        self.device = device
        self.policy_net = create_net(observation_shape, n_agents, n_actions, **policy_args).to(device)
        self.target_net = create_net(observation_shape, n_agents, n_actions, **policy_args).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters(), **opt_args)
        self.gamma = gamma
        self.n_agents = n_agents

    def init_episode(self, observations):
        self.hidden = self.policy_net.init_hidden(1)

    # def get_action(self, observations):
    #     observations = observations.unsqueeze(0)
    #     with th.no_grad():
    #         actions = self.policy_net(observations).max(-1)[1].view(-1)

    def get_q(self, observations):
        with th.no_grad():
            q, h = self.policy_net(observations.unsqueeze(0), self.hidden)
            self.hidden = h
            return q.squeeze(0)

    def optimize(self, prev_observations, actions, observations, rewards, done, valid):
        seq_length, batch_size, n_agents = actions.shape

        # this will not work with multi yet
        hidden = self.policy_net.init_hidden(batch_size)
        policy_q_values, hidden = self.policy_net(prev_observations, hidden)

        policy_state_action_values = policy_q_values.gather(-1, actions.unsqueeze(-1))

        # next_state_values = th.zeros((seq_length,batch_size, n_agents), device=self.device)
        hidden = self.target_net.init_hidden(batch_size)
        target_next_q_values, hidden = self.target_net(observations, hidden)
        next_state_values = target_next_q_values.max(-1)[0].detach()
        # next_state_values = self.target_net(observations, hidden).max(-1)[0].detach()

        # next_state_values[~done] = self.target_net(observations).max(2)[0].detach()

        import ipdb; ipdb.set_trace()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + rewards

        # Compute Huber loss
        loss = F.smooth_l1_loss(
            policy_state_action_values, expected_state_action_values.unsqueeze(-1), reduction='none')

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def log(self, writer):
        self.policy_net.log(writer, prefix='policyNet')