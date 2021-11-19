"""
Graph Coloring Environment
"""

import torch as th
import numpy as np
import collections
import os
from .utils.graph import create_graph
from ..utils.io import ensure_dir
from ..utils.utils import int_to_string
from aci.envs.reward_metrics import calc_metrics, calc_reward, create_reward_vec, METRIC_NAMES


def create_map(n_nodes, mapping_type=None,  **_):
    if mapping_type == 'random':
        return th.randperm(n_nodes)
    elif mapping_type == 'fixed':
        return th.arange(n_nodes)
    else:
        raise ValueError('Unkown mapping type.')


def create_maps(agent_type_args, n_nodes):
    """
    Mapping between agent_idx and network position.

    Returns:
        mapping: A tensor of shape ``
    """
    return th.stack([
        create_map(**ata, n_nodes=n_nodes)
        for ata in agent_type_args.values()
    ], dim=-1)


def create_control(
        n_nodes, n_agent_types, n_control=None, correlated=None, cross_correlated=None,
        device=None, **kwargs):
    """
    Args:
        control: p, t
    """
    if n_control is None:
        return None
    assert (n_nodes is not None)
    assert (correlated is not None)
    assert (cross_correlated is not None)
    if correlated and cross_correlated:
        c = th.randint(low=0, high=n_control, size=(1, 1), device=device)\
            .expand(n_nodes, n_agent_types)
    if correlated and not cross_correlated:
        c = th.randint(low=0, high=n_control, size=(1, n_agent_types),
                       device=device).expand(n_nodes, -1)
    if not correlated and cross_correlated:
        c = th.randint(low=0, high=n_control, size=(n_nodes, 1),
                       device=device).expand(-1, n_agent_types)
    if not correlated and not cross_correlated:
        c = th.randint(low=0, high=n_control, size=(
            n_nodes, n_agent_types), device=device)
    return c.unsqueeze(0).unsqueeze(0)


def pad_neighbors(neighbors, max_degree, device):
    assert max(len(
        n) for n in neighbors) <= max_degree, 'Networks has more neighbors then max neighbors.'

    padded_neighbors = [
        [i] + n + [-1]*(max_degree - len(n))
        for i, n in enumerate(neighbors)
    ]
    mask = [
        [False]*(len(n)+1) + [True]*(max_degree - len(n))
        for i, n in enumerate(neighbors)
    ]
    return (
        th.tensor(padded_neighbors, dtype=th.int64, device=device),
        th.tensor(mask, dtype=th.bool, device=device)
    )


class NetworkGame():
    """
    An multiagent environment. Agents are placed on a undirected network. On each
    node on of two types of agents are placed. On each timestep agents can select a
    discrete action. Reward is depending on whether actions of agents on the same
    node match and whether actions of neighboring actions match.

    The environment keeps track of historic agents and allows to resample from those.
    Periodically this history is saved, which allows to reconstruct the dynamic.


    Indices:
        t: agent types [0..1]
        p: positions / nodes [0..n_nodes]
        a: actions [0..n_actions]
        s: episode step [0..episode_steps (+ 1)]
        h: history position [0..max_history]
        b: batch idx [0..batch_size]
        n: neighbors [0..max_neighbors (+ 1)]
        m: metrics [0..len(metric_names)]
    """

    def __init__(
            self, *, n_nodes, n_actions, null_action=False, episode_steps, max_history,
            network_period=0, mapping_period=0, max_neighbors,
            reward_args, agent_type_args, graph_args, control_args,
            out_dir=None, save_interval, device):
        """
        Args:
            n_nodes: Number of nodes of the network.
            n_actions: Number of actions available to each agent.
            episode_steps: Number of steps for one episode.
            max_history: Maximum number of episodes to memorize.
            network_period: Period of episodes to resample a new network.
            mapping_period: Period of episodes to remap agents between the two types.
            agent_type_args: Settings for each agent type.
            control_args: Settings for the control integer.
            graph_args: Settings for the network creation.
            out_dir: Folder to store data.
            device: A torch device.
        """
        self.device = device

        self.mapping_period = mapping_period
        self.network_period = network_period
        self.max_history = max_history
        self.graph_args = graph_args
        self.agent_types = list(agent_type_args.keys())
        self.agent_types_idx = {v: i for i, v in enumerate(self.agent_types)}
        self.agent_type_args = agent_type_args
        self.metric_names = METRIC_NAMES

        self.episode_steps = episode_steps
        self.n_nodes = n_nodes
        self.n_actions = n_actions
        self.null_action = null_action
        self.n_agent_types = len(self.agent_types)
        self.control_args = control_args
        self.n_control = self.control_args.get('n_control', 0)
        # maximum number of neighbors
        self.max_neighbors = max_neighbors
        # self.max_neighbors = determine_max_degree(
        #     n_nodes=n_nodes, **graph_args)

        self.episode = -1
        self.out_dir = out_dir
        if out_dir is not None:
            ensure_dir(out_dir)
        self.save_interval = save_interval
        self.init_history()

        self.reward_vec = create_reward_vec(reward_args, self.device)

        self.info = {
            'n_agent_types': self.n_agent_types,
            'n_actions': self.n_actions,
            'n_nodes': self.n_nodes,
            'n_control': self.n_control,
            'max_neighbors': self.max_neighbors,
            'episode_steps': self.episode_steps,
            'metric_names': self.metric_names,
            'actions': [int_to_string(i) for i in range(self.n_actions)],
            'agents': [int_to_string(i) for i in range(self.n_nodes)],
            'agent_types': self.agent_types,
            'metric_names': self.metric_names,
        }

    # untested
    def init_history(self):
        self.history_queue = collections.deque([], maxlen=self.max_history)
        self.step_history = {
            'actions': th.empty((self.max_history, self.episode_steps + 1, self.n_nodes,
                                 self.n_agent_types), dtype=th.int64, device=self.device),  # h s+ p t
            'reward': th.empty((self.max_history, self.episode_steps + 1, self.n_nodes,
                                self.n_agent_types, ), dtype=th.float32, device=self.device),  # h s p t
            'metrics': th.empty((self.max_history, self.episode_steps + 1, self.n_nodes,
                                 self.n_agent_types, len(self.metric_names)),
                                dtype=th.float, device=self.device),  # h s+ p t m
            'control_int': th.empty(
                (self.max_history, self.episode_steps + \
                 1, self.n_nodes, self.n_agent_types),
                dtype=th.int64, device=self.device)  # h s+ p
        }
        self.episode_history = {
            'episode': th.empty(
                (self.max_history,), dtype=th.int64, device=self.device),  # h
            'neighbors': th.empty(
                (self.max_history, self.n_nodes, self.max_neighbors + 1), dtype=th.int64, device=self.device),  # h p n+
            'neighbors_mask': th.empty(
                (self.max_history, self.n_nodes, self.max_neighbors + 1), dtype=th.bool, device=self.device),  # h p n+
            'agent_map': th.empty(
                (self.max_history, self.n_nodes, self.n_agent_types), dtype=th.int64, device=self.device)  # h p
        }

    def init_episode(self):
        self.episode += 1
        self.episode_step = -1
        if self.episode != 0:
            prev_hist_idx = self.current_hist_idx
        self.current_hist_idx = self.episode % self.max_history
        episode_state = {}

        # create new graph (either only on first episode, or every network_period episode)
        if (self.episode == 0) or (
                (self.network_period > 0) and (self.episode % self.network_period == 0)):
            neighbors = create_graph(n_nodes=self.n_nodes, **self.graph_args)
            neighbors, neighbors_mask = pad_neighbors(
                neighbors, self.max_neighbors, self.device)
        else:
            neighbors = self.episode_history['neighbors'][prev_hist_idx]
            neighbors_mask = self.episode_history['neighbors_mask'][prev_hist_idx]

            # mapping between network and agents
        if (self.episode == 0) or (
                (self.mapping_period > 0) and (self.episode % self.mapping_period == 0)):
            agent_map = create_maps(self.agent_type_args, self.n_nodes)
        else:
            agent_map = self.episode_history['agent_map'][prev_hist_idx]

        episode_state = {
            'neighbors': neighbors,
            'neighbors_mask': neighbors_mask,
            'agent_map': agent_map,
            'episode': self.episode
        }

        self.history_queue.appendleft(self.current_hist_idx)
        # add episode attributes to history
        for k, v in episode_state.items():
            self.episode_history[k][self.current_hist_idx] = v

        # random init action
        if self.null_action:
            init_actions = th.zeros((1, 1, self.n_nodes, self.n_agent_types),
                                    dtype=th.int64, device=self.device)
        else:
            init_actions = th.tensor(
                np.random.randint(0, self.n_actions,
                                  (1, 1, self.n_nodes, self.n_agent_types)),
                dtype=th.int64, device=self.device)

        # init step
        return self.step(init_actions)

    def step(self, actions):
        self.episode_step += 1

        assert actions.max() <= self.n_actions - 1
        assert actions.dtype == th.int64

        if (self.episode_step == self.episode_steps):
            done = True
        elif self.episode_step > self.episode_steps:
            raise ValueError('Environment is done already.')
        else:
            done = False

        episode_state = {
            k: v[[self.current_hist_idx]]
            for k, v in self.episode_history.items()
            if v is not None
        }
        neighbors = episode_state['neighbors']
        neighbors_mask = episode_state['neighbors_mask']
        metrics = calc_metrics(
            actions, neighbors, neighbors_mask, self.null_action)  # h s+ p m t
        reward = calc_reward(metrics, self.reward_vec)
        control_int = create_control(
            n_nodes=self.n_nodes, n_agent_types=self.n_agent_types, **self.control_args,
            device=self.device)  # h s+ p t

        step_state = {
            'metrics': metrics,
            'reward': reward,
            'actions': actions,
            'control_int': control_int
        }

        # add step attributes to history
        for k, v in step_state.items():
            self.step_history[k][self.current_hist_idx,
                                 self.episode_step] = v[0, 0]

        state = {**step_state, **episode_state}
        return state, reward,  done

    def from_history(self, hist_idx):
        return {
            **{k: v[hist_idx] for k, v in self.episode_history},
            **{k: v[hist_idx] for k, v in self.step_history},
        }

    def get_current(self):
        return {
            **{k: v[[self.current_hist_idx]] for k, v in self.episode_history},
            **{k: v[[self.current_hist_idx], [self.episode_step]] for k, v in self.step_history},
        }

    def write_history(self):
        if self.out_dir is not None:
            filename = os.path.join(self.out_dir, f"{self.episode + 1}.pt")
            chi = self.current_hist_idx + 1
            th.save(
                {
                    'agent_types': self.agent_types,
                    'actions': [int_to_string(i) for i in range(self.n_actions)],
                    'agents': [int_to_string(i) for i in range(self.n_nodes)],
                    'metric_names': self.metric_names,
                    **{k: v[:chi:self.save_interval] for k, v in self.episode_history.items()},
                    **{k: v[:chi:self.save_interval] for k, v in self.step_history.items()},
                },
                filename
            )

    def finish_episode(self):
        if self.current_hist_idx == self.max_history - 1:
            self.write_history()

    def sample(self, batch_size, agent_type, horizon=None, last=False,  **kwarg):
        if batch_size > len(self.history_queue):
            return

        agent_type_idx = self.agent_types_idx[agent_type]

        if not last:
            assert horizon is not None
            assert horizon <= self.max_history
            eff_horizon = min(len(self.history_queue), horizon)
            pos_idx = np.random.choice(eff_horizon, batch_size)
        else:
            assert batch_size <= self.max_history
            # get the last n episodes (n=batch_size)
            pos_idx = np.arange(batch_size)

        hist_idx = th.tensor([self.history_queue[pidx]
                              for pidx in pos_idx], dtype=th.int64)

        keys = ['actions', 'metrics', 'control_int',
                'neighbors', 'neighbors_mask', 'agent_map']

        state = {
            **{k: v[hist_idx] for k, v in self.episode_history.items() if k in keys},
            **{k: v[hist_idx] for k, v in self.step_history.items() if k in keys},
        }
        actions = self.step_history['actions'][hist_idx].select(-1, agent_type_idx)[
            :, 1:]
        reward = self.step_history['reward'][hist_idx].select(-1, agent_type_idx)[
            :, 1:]

        return state, actions, reward

    # TODO: Unit test.

    def __del__(self):
        if hasattr(self, 'current_hist_idx'):
            if (self.current_hist_idx != (self.max_history - 1)):
                self.write_history()
