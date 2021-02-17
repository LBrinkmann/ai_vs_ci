"""
Graph Coloring Environment
"""

import torch as th
import numpy as np
import collections
import string
import os
from .utils.graph import create_graph, determine_max_degree, pad_neighbors
from ..utils.io import ensure_dir
from aci.envs.reward_metrics import calc_metrics, calc_reward, create_reward_vec


def create_map(mapping_type, n_nodes, **_):
    if mapping_type == 'random':
        th.randperm(n_nodes)
    else:
        return th.arange(n_nodes)


def create_maps(agent_types, agent_type_args, n_nodes, episode):
    """
    Mapping between agent_idx and network position.
    """
    return th.stack([
        create_map(**agent_type_args[at], n_nodes=n_nodes)
        for at in agent_types
    ])


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
        c = th.randint(low=0, high=n_control, size=(n_nodes, n_agent_types), device=device)
    return c


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
            self, *, n_nodes, n_actions, episode_steps, max_history,
            network_period, mapping_period, reward_period,
            rewards_metrics, graph_args, secrete_args={},
            out_dir=None, device):
        """
        Args:
            n_nodes: Number of nodes of the network.
            n_actions: Number of actions available to each agent.
            episode_steps: Number of steps for one episode.
            max_history: Maximum number of episodes to memorize.
            network_period: Period of episodes to resample a new network.
            mapping_period: Period of episodes to remap agents between the two types.
            reward_period: Period of steps to reward agents.
            rewards_metrics: Settings for calculation of rewards.
            graph_args: Settings for the network creation.
            secrete_args: Settings for calculation of secrets.
            out_dir: Folder to store data.
            device: A torch device.
        """

        self.metric_names = [
            'ind_coordination', 'avg_coordination', 'local_coordination', 'local_catch',
            'ind_catch', 'avg_catch', 'rewarded'
        ]

        self.episode_steps = episode_steps
        self.mapping_period = mapping_period
        self.network_period = network_period
        self.reward_period = reward_period
        self.max_history = max_history
        self.graph_args = graph_args
        self.secrete_args = secrete_args
        self.agent_types = ['ci', 'ai']
        self.agent_types_idx = {'ci': 0, 'ai': 1}
        self.n_nodes = n_nodes
        self.n_actions = n_actions
        self.n_agent_types = len(self.agent_types)
        self.n_control = self.control_args.get('n_control', 0)

        self.episode = -1
        # maximum number of neighbors
        self.max_neighbors = determine_max_degree(n_nodes=n_nodes, **graph_args)
        self.out_dir = out_dir
        self.device = device
        if out_dir is not None:
            ensure_dir(out_dir)
        self.init_history()

        self.reward_vec = create_reward_vec(self.agent_types, self.device, rewards_metrics)

        self.info = {
            'n_agent_types': self.n_agent_types,
            'n_actions': self.n_actions,
            'n_nodes': self.n_nodes,
            'n_control': self.n_control,
            'episode_steps': self.episode_steps,
            'metric_names': self.metric_names,
            'actions': [string.ascii_uppercase[i] for i in range(self.n_actions)],
            'agents': [string.ascii_uppercase[i] for i in range(self.n_nodes)],
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
                (self.max_history, self.episode_steps + 1, self.n_nodes), dtype=th.int64, device=self.device)  # h s+ p
        }
        self.episode_history = {
            'episode': th.empty(
                (self.max_history,), dtype=th.int64, device=self.device),  # h
            'neighbors': th.empty(
                (self.max_history, self.n_nodes, self.max_neighbors + 1), dtype=th.int64, device=self.device),  # h p n+
            'neighbors_mask': th.empty(
                (self.max_history, self.n_nodes, self.max_neighbors + 1), dtype=th.bool, device=self.device),  # h p n+
            'agent_map': th.empty(
                (self.max_history, self.n_nodes), dtype=th.int64, device=self.device)  # h p
        }

    def init_episode(self):
        self.episode += 1
        self.episode_step = -1
        self.current_hist_idx = self.episode % self.max_history

        # create new graph (either only on first episode, or every network_period episode)
        if (self.episode == 0) or (
                (self.network_period > 0) and (self.episode % self.network_period == 0)):
            self.graph, neighbors, adjacency_matrix, self.graph_info = create_graph(
                n_nodes=self.n_nodes, **self.graph_args)
            self.neighbors, self.neighbors_mask = pad_neighbors(neighbors, self.max_neighbors)

        # mapping between network and agents
        if (self.episode == 0) or (
                (self.network_period > 0) and (self.episode % self.network_period == 0)):
            self.agent_map = create_maps(
                self.agent_types, self.agent_type_args, self.n_nodes, self.episode)

        self.history_queue.appendleft(self.current_hist_idx)

        # add episode attributes to history
        for k in self.episode_history.keys():
            self.episode_history[k][self.current_hist_idx] = self.__dict__[k]

        # random init action
        init_actions = th.tensor(
            np.random.randint(0, self.n_actions, (self.n_nodes, self.n_agent_types)),
            dtype=th.int64, device=self.device)

        # init step
        return self.step(init_actions)

    def step(self, actions):
        self.episode_step += 1

        assert actions.max() <= self.n_actions - 1
        assert actions.dtype == th.int64

        if (self.episode_step + 1 == self.episode_steps):
            done = True
        elif self.episode_step >= self.episode_steps:
            raise ValueError('Environment is done already.')
        else:
            done = False

        self.actions = actions
        _actions = self.actions.unsqueeze(0).unsqueeze(0)
        _neighbors = self.neighbors.unsqueeze(0)
        _neighbors_mask = self.neighbors_mask.unsqueeze(0)

        _metrics = calc_metrics(_actions, _neighbors, _neighbors_mask)  # h s+ p m t

        _reward = calc_reward(_metrics, self.reward_vec)  # h s+ p t
        self.metrics = _metrics.squeeze(0).squeeze(0)
        self.reward = _reward.squeeze(0).squeeze(0)

        self.control_int = create_control(
            n_nodes=self.n_nodes, n_agents=self.n_agents,
            **self.control_args, device=self.device)  # p t

        # add step attributes to history
        for k in self.step_history.keys():
            self.step_history[k][self.current_hist_idx, self.episode_step] = self.__dict__[k]

        state = {
            self.__dict__[k]
            for k in self.step_history.keys()
        }
        return state, self.reward,  done

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
                    'actions': [string.ascii_uppercase[i] for i in range(self.n_actions)],
                    'agents': [string.ascii_uppercase[i] for i in range(self.n_nodes)],
                    'metric_names': self.metric_names,
                    **{k: v[:chi]for k, v in self.episode_history},
                    **{k: v[:chi]for k, v in self.step_history},
                },
                filename
            )

    def finish_episode(self):
        if self.current_hist_idx == self.max_history - 1:
            self.write_history()

    def observe(self):
        keys = ['actions', 'metrics', 'control_int', 'neighbors', 'neighbors_mask', 'agent_map']
        return {
            n: self.__dict__[n] for n in keys
        }

    def sample(self, mode, batch_size, horizon=None, last=False, **kwarg):
        if batch_size > len(self.history_queue):
            return

        if not last:
            assert horizon is not None
            assert horizon <= self.max_history
            eff_horizon = min(len(self.history_queue), horizon)
            pos_idx = np.random.choice(eff_horizon, batch_size)
        else:
            assert batch_size <= self.max_history
            # get the last n episodes (n=batch_size)
            pos_idx = np.arange(batch_size)

        hist_idx = th.tensor([self.history_queue[pidx] for pidx in pos_idx], dtype=th.int64)

        keys = ['actions', 'metrics', 'control_int', 'neighbors', 'neighbors_mask', 'agent_map']

        return {
            **{k: v[hist_idx] for k, v in self.episode_history if k in keys},
            **{k: v[hist_idx] for k, v in self.step_history if k in keys},
        }

    # TODO: Unit test.

    def __del__(self):
        if self.current_hist_idx != self.max_history - 1:
            self.write_history()
