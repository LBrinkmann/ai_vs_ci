"""
Graph Coloring Environment
"""

import random
import torch as th
import numpy as np
import matplotlib as mpl
import string
mpl.use('Agg')
import matplotlib.pyplot as plt


class TableGame():
    def __init__(self, n_agents, max_steps, rewards, device):
        self.steps = 0
        self.max_steps = max_steps
        self.new_graph()
        self.agent_groups = rewards.keys()
        self.rewards = rewards


    def step(self, actions, writer=None):
        ai_rewards = [self.rewards['ai'][ai][ci] for ai, ci in zip(actions['ai'], actions['ci'])]
        ci_rewards = [self.rewards['ci'][ci][ai] for ai, ci in zip(actions['ai'], actions['ci'])]

        if (self.steps == self.max_steps):
            done = True
        elif self.steps > self.max_steps:
            raise ValueError('Environment is done already.')
        else:
            done = False
        self.steps += 1

        observations = {
            'ai': th.tensor([[ai, ci] for ai, ci in zip(actions['ai'], actions['ci'])]),
            'ci': th.tensor([[ci, ai] for ai, ci in zip(actions['ai'], actions['ci'])])
        }

        rewards = {
            'ai': th.tensor(ai_rewards),
            'ci': th.tensor(ci_rewards)
        }


        if writer:
            self._log(actions, observations, rewards, done, writer) 

        return observations, rewards, done, {}

    def reset(self, init=False):
        self.steps = 0

        self.agg_metrics = {}

        random_actions = th.random.randint(1, shape=(self.n_agents, 2))

        observations = {
            'ai': random_actions,
            'ci': random_actions[:, ::-1]
        }

        return observations

    def close(self):
        pass

    def _log(self, actions, observations, metrics, done, writer):
        if len(self.agg_metrics) == 0:
            self.agg_metrics = metrics
        else:
            self.agg_metrics = {k: self.agg_metrics[k] + v for k, v in metrics.items()}

        if done and writer.check_on(on='final'):
            writer.add_metrics(
                'final',
                {
                    'avg_ci_rewards': self.agg_metrics['ci'].mean(),
                    'avg_ai_rewards': self.agg_metrics['ai'].mean(),
                },
                {},
                tf=[],
                on='final'
            )




