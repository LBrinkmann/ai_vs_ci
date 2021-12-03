import torch as th
import numpy as np


def onehot(x, size, device):
    oh = th.zeros(*x.shape, size, dtype=th.float32, device=device)
    oh.scatter_(-1, x.unsqueeze(-1), 1)
    return oh


def binary(x, bits):
    mask = 2**th.arange(bits).flip(0).to(x.device, x.dtype)
    bits = x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()
    return bits


class ViewEncoder():
    def __init__(self, *, actions_view=None, metric_view=None,
                 control_view=None, env_info, device, agent_type=None):
        self.n_actions = env_info['n_actions']
        self.device = device
        n_control = env_info['n_control']
        agent_types = env_info['agent_types']
        metric_names = env_info['metric_names']

        self.shape = 0
        if actions_view is not None:
            self.at_idx = [i for i, n in enumerate(agent_types) if n in actions_view]
            self.shape += len(actions_view) * self.n_actions
        else:
            self.at_idx = None

        if metric_view is not None:
            self.metric_at_idx = [agent_types.index(vm['agent_type']) for vm in metric_view]
            self.metric_m_idx = [metric_names.index(vm['metric_name']) for vm in metric_view]
            self.shape += len(metric_view)
        else:
            self.metric_at_idx = None
            self.metric_m_idx = None

        if control_view is not None:
            self.control_size = int(np.log2(n_control + 1))
            self.control_at_idx = agent_types.index(control_view)
            self.shape += self.control_size
        else:
            self.control_size = None
            self.control_at_idx = None

    def __call__(self, actions, metrics, control_int, **_):
        """
        Args:
            actions: h s+ p t
            metrics: h s+ p t m
            control_int: h s+ p t
        return:
            view: h s+ p i
        """
        h, s, p, t = actions.shape
        a = self.n_actions

        views = []
        if self.at_idx is not None:
            actions_view = onehot(actions[:, :, :, self.at_idx], a, self.device)  # h s+ p t a
            actions_view = actions_view.reshape(h, s, p, a*len(self.at_idx))  # h s+ p i'
            views.append(actions_view)

        if self.metric_at_idx is not None:
            metric_view = metrics[:, :, :, self.metric_at_idx, self.metric_m_idx]  # h s+ p i''
            views.append(metric_view)

        if self.control_size is not None:
            control_view = binary(
                control_int[:, :, :, self.control_at_idx], self.control_size)  # h s+ p i'''
            views.append(control_view)
        if len(views) > 0:
            view = th.cat(views, dim=-1)  # h s+ p i
        else:
            view = None
        return {'view': view}
