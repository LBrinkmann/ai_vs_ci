# derived from https://github.com/oxwhirl/pymarl/blob/master/src/components/action_selectors.py


import torch as th
from torch.distributions import Categorical
import math
import random


class MultiActionSelector():
    def __init__(self, device, **args):
        self.args = args
        self.steps_done = 0
        self.device = device
        self.eps = self.calc_eps(steps_done=0, **self.args)

    @staticmethod
    def calc_eps(eps_end, eps_start, eps_decay, steps_done):
        eps_threshold = eps_end + (eps_start - eps_end) * \
            math.exp(-1. * steps_done / eps_decay)
        return eps_threshold

    def info(self):
        return {'eps': self.eps}

    def select_action(self, proposed_action, avail_actions=None, test_mode=False):
        if test_mode:
            eps = 0
        else:
            eps = self.eps
            self.steps_done += 1
            self.eps = self.calc_eps(steps_done=self.steps_done, **self.args)

        if avail_actions is None:
            avail_actions = th.ones_like(proposed_action)

        masked_q_values = proposed_action.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(proposed_action[:, :, 0])
        pick_random = (random_numbers < eps).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions

    def log(self, writer, step, details):
        writer.add_scalar('actionSelector.eps', self.eps, step)


class ActionSelector():
    def __init__(self, device, **args):
        self.args = args
        self.steps_done = 0
        self.device = device

    @staticmethod
    def calc_eps(eps_end, eps_start, eps_decay, steps_done):
        eps_threshold = eps_end + (eps_start - eps_end) * \
            math.exp(-1. * steps_done / eps_decay)
        return eps_threshold

    def select_action(self, proposed_action):
        eps = ActionSelector.calc_eps(steps_done=self.steps_done, **self.args)
        self.steps_done += 1
        self.last_eps = eps
        if random.random() > eps:
            return proposed_action
        else:
            return th.tensor([[random.randrange(len(proposed_action))]], device=self.device, dtype=th.long)

    def log(self, writer, step, details):
        writer.add_scalar('actionSelector.eps', self.last_eps, step)
