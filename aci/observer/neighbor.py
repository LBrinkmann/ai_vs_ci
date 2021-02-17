import torch as th
import numpy as np
from .reward_metrics import project_on_neighbors


def onehot(x, size, device):
    oh = th.zeros(*x.shape, size, dtype=th.float32, device=device)
    oh.scatter_(-1, x.unsqueeze(-1), 1)
    return oh


def binary(x, bits):
    mask = 2**th.arange(bits).to(x.device, x.dtype)
    return (x+1).unsqueeze(-1).bitwise_and(mask).ne(0).byte()


def view_encoder(
        state, metrics, control_int, n_actions, n_control, device, agent_types,
        metric_names, view_control_agent_type, state_view, metric_view, control_view, **_other):
    """
    Args:
        state: h s+ p t
        metrics: h s+ p t m
        control_int: h s+ p t
    return:
        view: h s+ p i
    """
    h, s, p, t = state.shape
    a = n_actions

    views = []

    if state_view is not None:
        at_idx = [i for i, n in enumerate(agent_types) if n in state_view]
        state_view = onehot(state, n_actions, device)  # h s+ p t a
        state_view = state_view[:, :, :, at_idx].reshape(h, s, p, -1, a)  # h s+ p i'
        views.append(state_view)

    if metric_view is not None:
        metric_at_idx = [agent_types.index(vm['agent_type']) for vm in metric_view]
        metric_m_idx = [agent_types.index(vm['metric_names']) for vm in metric_view]
        metric_view = metrics[:, :, :, metric_at_idx, metric_m_idx]  # h s+ p i''
        views.append(metric_view)

    if control_view is not None:
        control_size = int(np.log2(n_control + 1))
        control_at_idx = agent_types.index(control_view)
        control_view = binary(control_int[:, :, :, control_at_idx], control_size)  # h s+ p i'''
        views.append(control_view)

    view = th.cat(views, dim=-1)  # h s+ p i
    return view

# metric_names, view_control_agent_type, state_view, metric_view, control_view,


class ViewEncoder():
    def __init__(self, state_view, metric_view, control_view, env_info, device):
        self.n_actions = env_info['n_actions']
        self.device = device
        n_control = env_info['n_control']
        agent_types = env_info['agent_types']

        self.shape = 0
        if state_view is not None:
            self.at_idx = [i for i, n in enumerate(agent_types) if n in state_view]
            self.shape += len(state_view) * self.n_actions
        else:
            self.at_idx = None

        if metric_view is not None:
            self.metric_at_idx = [agent_types.index(vm['agent_type']) for vm in metric_view]
            self.metric_m_idx = [agent_types.index(vm['metric_names']) for vm in metric_view]
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

    def __call__(self, state, metrics, control_int, **_):
        """
        Args:
            state: h s+ p t
            metrics: h s+ p t m
            control_int: h s+ p t
        return:
            view: h s+ p i
        """
        h, s, p, t = state.shape
        a = self.n_actions

        views = []
        if self.at_idx is not None:
            state_view = onehot(state[:, :, :, self.at_idx], a, self.device)  # h s+ p t a
            state_view = state_view.reshape(h, s, p, -1, a)  # h s+ p i'
            views.append(state_view)

        if self.metric_at_idx is not None:
            metric_view = metrics[:, :, :, self.metric_at_idx, self.metric_m_idx]  # h s+ p i''
            views.append(metric_view)

        if self.control_size is not None:
            control_view = binary(
                control_int[:, :, :, self.control_at_idx], self.control_size)  # h s+ p i'''
            views.append(control_view)

        view = th.cat(views, dim=-1)  # h s+ p i
        return view


class NeighborView():
    def __init__(self, agent_type, neighbor_view_args, control_view_args, env_info, device):
        self.device = device
        self.agent_type = agent_type
        self.agent_type_idx = env_info['agent_types'].index(agent_type)

        self.neighbor_encoder = ViewEncoder(**neighbor_view_args, env_info=env_info, device=device)
        self.control_encoder = ViewEncoder(**control_view_args, env_info=env_info, device=device)

        n_nodes = env_info['n_nodes']
        max_neighbors = env_info['max_neigbors']

        self.shape = {
            'view': (-1, -1, n_nodes, max_neighbors, self.neighbor_encoder.shape),
            'control': (-1, -1, n_nodes, self.control_encoder.shape),
            'mask': (-1, -1, n_nodes, max_neighbors),
            'agent_map': (-1, -1, n_nodes),
            'control_int': (-1, -1, n_nodes)
        }

    def __call__(self, **state):
        h, s, p, t = state['state'].shape
        agent_type_idx = self.agent_type_idx

        view = self.neighbor_encoder(**state)  # h s+ p i
        view = project_on_neighbors(view, state['neighbor'], state['neighbor_mask'])  # h s+ p n i
        control = self.control_encoder(**state)  # h s+ p c

        mask = state['neighbor_mask'].unsqueeze(1).expand(-1, s, -1, -1)  # h s+ p n
        agent_map = state['agent_map'][:, :, :, agent_type_idx].unsqueeze(
            1).expand(-1, s, -1)  # h s+ p
        control_int = state['control_int'][:, :, :, agent_type_idx]  # h s+ p
        return {
            'view': view,   # h s+ p n i
            'control': control,  # h s+ p c
            'mask': mask,  # h s+ p n
            'agent_map': agent_map,  # h s+ p
            'control_int': control_int  # h s+ p
        }
