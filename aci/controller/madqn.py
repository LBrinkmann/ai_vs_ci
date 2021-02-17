import torch as th
import numpy as np
from aci.neural_modules import NETS
from .mawrapper import MultiAgentWrapper


class MADQN:
    def __init__(
            self, observation_shapes, n_agents, n_actions, model_args, opt_args, sample_args,
            agent_type, gamma, batch_size, target_update_freq, mix_freq=None, device):
        self.device = device
        self.policy_net = MultiAgentWrapper(
            observation_shapes, n_agents, n_actions, device=device, **model_args).to(device)
        self.target_net = MultiAgentWrapper(
            observation_shapes, n_agents, n_actions, device=device, **model_args).to(device)

        self.target_net.eval()
        self.optimizer = th.optim.RMSprop(self.policy_net.parameters(), **opt_args)
        self.gamma = gamma
        self.batch_size = batch_size
        self.sample_args = sample_args
        self.target_update_freq = target_update_freq
        self.mix_freq = mix_freq

    def get_q(self, observations):
        with th.no_grad():
            _observations = {
                k: v.unsqueeze(0).unsqueeze(0) if i is not None else None
                for k, v in observations.items()
            }
            return self.policy_net(**_observations).squeeze(0).squeeze(0)

    def update(self, observations, actions, rewards, episode):
        self.pre_optimise(episode)

        if training and done:
            sample = sampler(**self.sample_args)
            if sample is not None:
                self.pre_optimise(episode, training)
                observations = observer(sample)
                self.optimize(observations, sample['actions'])

    def pre_optimise(self, episode):
        self.policy_net.reset()
        if (self.mix_freq is not None) and (episode % self.mix_freq == 0):
            self.policy_net.mix_weights()

        if (episode % self.target_update_freq == 0):
            # copy policy net to target net
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def optimize(self, observations, actions, rewards):
        previous_obs, current_obs = shift_obs(observations)

        self.policy_net.reset()
        self.target_net.reset()

        policy_state_action_values = self.policy_net(
            **previous_obs).gather(-1, actions.unsqueeze(-1))

        next_state_values = th.zeros_like(rewards, device=self.device)
        next_state_values = self.target_net(**current_obs).max(-1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + rewards

        # Compute Huber loss
        loss = th.nn.functional.smooth_l1_loss(policy_state_action_values,
                                               expected_state_action_values.unsqueeze(-1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def log(self, writer):
        self.policy_net.log(writer, prefix='policyNet')
