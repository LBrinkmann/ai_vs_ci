# derived from https://github.com/oxwhirl/pymarl/blob/master/src/components/action_selectors.py


import torch as th
from torch.distributions import Categorical
import math
import random

def select_action(proposed_actions, eps):

    avail_actions = th.ones_like(proposed_actions)

    masked_q_values = proposed_actions
    masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

    random_numbers = th.rand_like(proposed_actions[:, 0])
    pick_random = (random_numbers < eps).long()
    random_actions = Categorical(avail_actions.float()).sample().long()
    picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=-1)[1]
    return picked_actions


class Scheduler():
    def __init__(self, phases, eval_period, eval_setting, episodes):
        self.phases = {p['episode']: p['setting'] for p in phases}
        self.eval_period = eval_period
        self.eval_setting = eval_setting
        self.episode = 0
        self.episodes = episodes
        self.last_eval = -1

    def __iter__(self):
        return self

    def __next__(self):
        if self.episode >= self.episodes:
            raise StopIteration()

        if ((self.episode % self.eval_period) == 0) & (self.episode > self.last_eval):
            obj = {
                **self.eval_setting,
                'mode': 'eval',
                'episode': self.episode
            }
            self.last_eval = self.episode
        else:
            if self.episode in self.phases:
                self.train_setting = self.phases[self.episode]
                print(f'Update train setting: {self.train_setting}')
            obj = {
                **self.train_setting,
                'mode': 'train',
                'episode': self.episode
            }
            self.episode = self.episode + 1
        return obj