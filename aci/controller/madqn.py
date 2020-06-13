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
import torch.nn as nn


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



class SharedWeightModel(nn.Module):
    def __init__(self, model):
        super(SharedWeightModel, self).__init__()
        self.model = model

    def forward(self, x):
        batch_size = x.shape[0]
        n_agents = x.shape[1]
        _x = x.reshape(-1, *x.shape[2:])
        y = self.model(_x)
        y = y.reshape(batch_size, n_agents, *y.shape[1:])
        return y
    
    def log(self, *args, **kwargs):
        self.model.log(*args, **kwargs)


net_types = {
    'shared_medium_conv': [FloatTransformer, SharedWeightModel, MediumConv],
    'shared_logistic': [FloatTransformer, SharedWeightModel, LogisticRegression],
    'shared_learningheuristic': [SharedWeightModel, LearningHeuristic],
}


def create_net(observation_shape, n_agents, n_actions, net_type, multi_type, **kwargs):
    if net_type == 'medium_conv':
        model = MediumConv(*observation_shape, n_actions, **kwargs)
        model = FloatTransformer(model)
    elif net_type == 'logistic':
        model = LogisticRegression(*observation_shape, n_actions, **kwargs)
        model = FloatTransformer(model)
    elif net_type == 'learningheuristic':
        model = LearningHeuristic(*observation_shape, n_actions, **kwargs)
    else:
        raise NotImplementedError(f"Net type not found: {net_type}")
    
    if multi_type == 'shared_weights':
        model = SharedWeightModel(model)
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

    def get_action(self, observations):
        observations = observations.unsqueeze(0)
        with th.no_grad():
            return self.policy_net(observations).max(-1)[1].view(-1)

    def get_q(self, observations):
        with th.no_grad():
            return self.policy_net(observations)

    def optimize(self, prev_observations, actions, observations, rewards, done):
        batch_size, n_agents = actions.shape

        # this will not work with multi yet
        policy_state_action_values = self.policy_net(prev_observations).gather(2, actions.unsqueeze(2))

        next_state_values = th.zeros((batch_size, n_agents), device=self.device)
        next_state_values[~done] = self.target_net(observations[~done]).max(2)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + rewards

        # Compute Huber loss
        loss = F.smooth_l1_loss(policy_state_action_values, expected_state_action_values.unsqueeze(2))

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