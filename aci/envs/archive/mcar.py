import gym
import torch as th
import numpy as np


class CarWrapper():
    def __init__(self, device, n_agents, max_steps):
        self.envs = [gym.make('MountainCar-v0').unwrapped for i in range(n_agents)]
        self.device = device
        self.n_agents = {'car': n_agents}
        self.steps = 0
        self.max_steps = max_steps

    def step(self, actions, writer=None):
        actions = actions['car']
        rewards = th.zeros(self.n_agents['car'])
        dones = th.zeros(self.n_agents['car'], dtype=th.bool)
        observations = th.zeros((self.n_agents['car'], *self.observation_shape['car']), dtype=th.float32)
        for i, env in enumerate(self.envs):
            observation, reward, done, _ = env.step(actions[i].item())
            rewards[i] = reward
            dones[i] = done
            observations[i] = th.tensor(observation, dtype=th.float32)
        done = (self.steps > self.max_steps) or dones.any()

        self.steps += 1

        if writer:
            self._log(actions, observations, {'tot_rewards': sum(rewards)}, done, writer) 

        # how to handle done with multiple cars is not clear yeat
        return {'car': observations}, {'car': rewards}, done, {}
    
    def reset(self):
        observations = th.zeros((self.n_agents['car'], *self.observation_shape['car']), dtype=th.float32)
        self.steps = 0
        self.agg_metrics = []
        for i, env in enumerate(self.envs):
            observation = env.reset()
            observations[i] = th.tensor(observation, dtype=th.float32)
        return {'car': observations}

    @property
    def n_actions(self):
        return {'car': self.envs[0].action_space.n}

    @property
    def observation_shape(self):
        return {'car': self.envs[0].observation_space.shape}

    def __del__(self):
        for env in self.envs:
            env.close()  

    
    def _log(self, actions, observations, metrics, done, writer):
        if len(self.agg_metrics) == 0:
            self.agg_metrics = metrics
        else:
            self.agg_metrics = {k: self.agg_metrics[k] + v for k, v in metrics.items()}

        if done and writer.check_on(on='final'):
            writer.add_metrics(
                'final',
                {
                    'tot_rewards': self.agg_metrics['tot_rewards'],
                },
                {},
                tf=[],
                on='final'
            )
