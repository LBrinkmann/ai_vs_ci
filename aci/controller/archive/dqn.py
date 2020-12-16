from collections import namedtuple
from aci.controller.utils import ReplayMemory, Transition
import math
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from aci.neural_modules.medium_conv import MediumConv


class DQN:
    def __init__(
            self, observation_shape, n_actions, replay_memory, opt_args, device):
        self.n_actions = n_actions
        self.device = device
        self.policy_net = MediumConv(observation_shape[0], observation_shape[1], n_actions).to(device)
        self.target_net = MediumConv(observation_shape[0], observation_shape[1], n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(replay_memory)
        self.opt_args = opt_args

    def get_action(self, state):
        with torch.no_grad():
            return self.policy_net(state).max(1)[1].view(1, 1)

    def optimize(self):
        batch_size, gamma = self.opt_args['batch_size'], self.opt_args['gamma']

        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def push_transition(self, *transition):
        self.memory.push(*transition)

    def log(self, writer, step, details):
        if details:
            self.policy_net.log(writer, step, prefix='policyNet')
