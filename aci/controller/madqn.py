import torch as th
from .utils.shift_observations import shift_obs
from .mawrapper import MultiAgentWrapper
from aci.utils.memory import Memory


class MADQN():
    def __init__(
            self, *, observation_shape, env_info, wrapper_args, model_args, opt_args,
            gamma, sample_args, target_update_freq, agent_type, mix_freq=None, device):
        self.device = device

        self.policy_net = MultiAgentWrapper(
            observation_shape=observation_shape, env_info=env_info, device=device,
            model_args=model_args, **wrapper_args).to(device)
        self.target_net = MultiAgentWrapper(
            observation_shape=observation_shape, env_info=env_info, device=device,
            model_args=model_args, **wrapper_args).to(device)

        self.target_net.eval()
        self.optimizer = th.optim.RMSprop(self.policy_net.parameters(), **opt_args)
        self.gamma = gamma
        self.sample_args = sample_args
        self.target_update_freq = target_update_freq
        self.mix_freq = mix_freq
        self.agent_type = agent_type
        self.memory = Memory(
            episode_steps=env_info['episode_steps'] + 1, device=device,
            max_history=sample_args['horizon'])

    def init_episode(self, observation, action, reward, episode):
        self.memory.start_episode(episode)
        self.memory.add({**observation, 'action': action, 'reward': reward}, episode_step=0)

        if (self.mix_freq is not None) and (episode % self.mix_freq == 0):
            self.policy_net.mix_weights()
        if (episode % self.target_update_freq == 0):
            # copy policy net to target net
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.reset()
        self.target_net.reset()

    def get_q(self, **observations):
        with th.no_grad():
            return self.policy_net(**observations)

    def update(self, observation, action, reward, done, step, episode):
        self.memory.add(
            {**observation, 'action': action, 'reward': reward}, episode_step=step+1)

        if done and (len(self.memory) >= self.sample_args['batch_size']):
            sample = self.memory.get(**self.sample_args)

            previous_sample, current_sample = shift_obs(sample)
            previous_obs = {
                k: v for k, v in previous_sample.items() if k not in ['action', 'reward']
            }
            current_obs = {
                k: v for k, v in current_sample.items() if k not in ['action', 'reward']
            }
            actions = current_sample['action']
            rewards = current_sample['reward']

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
