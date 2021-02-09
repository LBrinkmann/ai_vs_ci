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
import aci.envs.transformer as trf


def show_agent_type_secrets(agent_type=None, agent_types=None, n_seeds=None, **kwargs):
    if (n_seeds is not None) and (agent_type is not None) and (agent_type in agent_types):
        return True
    else:
        return False


def get_secrets(n_nodes=None, n_seeds=None, correlated=None, device=None, **kwargs):
    if n_seeds is None:
        return None
    assert (n_nodes is not None)
    assert (correlated is not None)
    if correlated:
        return th.randint(low=0, high=n_seeds, size=(1,), device=device).repeat(n_nodes)
    else:
        return th.randint(low=0, high=n_seeds, size=(n_nodes,), device=device)


def get_secrets_shape(n_nodes, n_seeds=None, agent_type=None, agent_types=None, **kwargs):
    if (n_seeds is not None) and (agent_type in agent_types):
        return {'shape': (n_nodes,), 'maxval': n_seeds - 1}
    else:
        return None


def do_update_secrets(episode_step, secret_period=None, **kwars):
    return (secret_period is None) or (episode_step % secret_period == 0) or (episode_step == 0)


def gen_secrets(n_seeds=None, **kwargs):
    if (n_seeds is not None):
        return True
    else:
        return False


class NetworkGame():
    """
    An multiagent environment. Agents are placed on a undirected network. On each
    node on of two types of agents are placed. On each timestep agents can select a
    discrete action. Reward is depending on whether actions of agents on the same
    node match and whether actions of neighboring actions match.

    The environment keeps track of historic agents and allows to resample from those.
    Periodically this history is saved, which allows to reconstruct the dynamic.


    Indices:
        n: nodes [0..n_nodes]
        a: actions [0..n_actions]
        s: episode step [0..episode_steps]
        h: history position [0..max_history]
    """

    def __init__(
            self, *, n_nodes, n_actions, episode_steps, max_history,
            network_period, mapping_period, reward_period,
            rewards_args, graph_args, secrete_args={},
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
            rewards_args: Settings for calculation of rewards.
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
        self.rewards_args = rewards_args
        self.secrete_args = secrete_args
        self.agent_types = ['ci', 'ai']
        self.agent_types_idx = {'ci': 0, 'ai': 1}
        self.n_nodes = n_nodes
        self.n_actions = n_actions
        self.episode = -1
        # maximum number of neighbors
        self.max_neighbors = determine_max_degree(n_nodes=n_nodes, **graph_args)
        self.out_dir = out_dir
        self.device = device
        self.init_history()
        if out_dir is not None:
            ensure_dir(out_dir)

        self.info2 = {
            'n_agent_types': len(self.agent_types),
            'n_actions': self.n_actions,
            'n_nodes': self.n_nodes,
            'episode_steps': self.episode_steps,
            'metric_names': self.metric_names,
            'actions': [string.ascii_uppercase[i] for i in range(self.n_actions)],
            'agents': [string.ascii_uppercase[i] for i in range(self.n_nodes)],
            'agent_types': self.agent_types,
            'metric_names': self.metric_names,
        }

    # TODO: understand and document

    def _update_secrets(self):
        if do_update_secrets(episode_step=self.episode_step, **self.secrete_args):
            self.secretes = get_secrets(
                **self.secrete_args, n_nodes=self.n_nodes, device=self.device)

    # untested
    def init_history(self):
        self.history_queue = collections.deque([], maxlen=self.max_history)
        self.history = {
            'episode': th.empty((self.max_history,), dtype=th.int64, device=self.device),
            'state': th.empty(
                (len(self.agent_types), self.n_nodes, self.max_history, self.episode_steps + 1), dtype=th.int64, device=self.device),
            'reward': th.empty(
                (len(self.agent_types), self.n_nodes, self.max_history, self.episode_steps), dtype=th.float32, device=self.device),
            'neighbors': th.empty(
                (self.n_nodes, self.max_history, self.max_neighbors + 1), dtype=th.int64, device=self.device),
            'neighbors_mask': th.empty(
                (self.n_nodes, self.max_history, self.max_neighbors + 1), dtype=th.bool, device=self.device),
            'ci_ai_map': th.empty(
                (self.n_nodes, self.max_history), dtype=th.int64, device=self.device),
            'metrics': th.empty(
                (self.n_nodes, self.max_history, self.episode_steps + 1, len(self.metric_names)), dtype=th.float, device=self.device)
        }
        if gen_secrets(**self.secrete_args):
            self.history['secretes'] = th.empty(
                (self.n_nodes, self.max_history, self.episode_steps + 1), dtype=th.int64, device=self.device)

    # untested

    def init_episode(self):
        self.episode += 1
        self.episode_step = -1
        self.current_hist_idx = self.episode % self.max_history

        # create new graph (either only on first episode, or every network_period episode)
        if (self.episode == 0) or (
                (self.network_period > 0) and (self.episode % self.network_period == 0)):
            self.graph, neighbors, adjacency_matrix, self.graph_info = create_graph(
                n_nodes=self.n_nodes, **self.graph_args)
            padded_neighbors, neighbors_mask = pad_neighbors(neighbors, self.max_neighbors)
            self.neighbors = th.tensor(padded_neighbors, dtype=th.int64, device=self.device)
            self.neighbors_mask = th.tensor(neighbors_mask, dtype=th.bool, device=self.device)
            self.adjacency_matrix = th.tensor(
                adjacency_matrix, dtype=th.bool, device=self.device)

        # mapping between ai agents and ci agents (outward: ci_ai_map, incoming: ai_ci_map)
        # ci_ai_map: [2, 0, 1], ai agent 0 is ci agent 2
        if (self.mapping_period > 0) and (self.episode % self.mapping_period == 0):
            self.ai_ci_map = th.randperm(self.n_nodes)
            self.ci_ai_map = th.argsort(self.ai_ci_map)
        elif self.episode == 0:
            self.ai_ci_map = th.arange(self.n_nodes)
            self.ci_ai_map = th.argsort(self.ai_ci_map)

        # init state
        self.state = th.tensor(
            np.random.randint(0, self.n_actions, (2, self.n_nodes)), dtype=th.int64, device=self.device)
        self._update_secrets()

        ci_reward, ai_reward, metrics = self._get_rewards()
        self.metrics = metrics
        # add to history
        self.history_queue.appendleft(self.current_hist_idx)
        self.history['episode'][self.current_hist_idx] = self.episode
        self.history['state'][:, :, self.current_hist_idx, self.episode_step + 1] = self.state
        self.history['neighbors'][:, self.current_hist_idx] = self.neighbors
        self.history['neighbors_mask'][:, self.current_hist_idx] = self.neighbors_mask

        if 'secretes' in self.history:
            self.history['secretes'][:, self.current_hist_idx,
                                     self.episode_step + 1] = self.secretes
        self.history['metrics'][:, self.current_hist_idx, self.episode_step + 1] = self.metrics

        self.history['ci_ai_map'][:, self.current_hist_idx] = self.ci_ai_map

        self.info = {
            'graph': self.graph,
            'graph_info': self.graph_info,
            'ai_ci_map': self.ai_ci_map.tolist(),
            'ci_ai_map': self.ci_ai_map.tolist()
        }

    def get_history(self):
        names = ['state', 'reward', 'neighbors', 'neighbors_mask', 'ci_ai_map', 'metrics']
        if 'secretes' in self.history:
            names.append('secretes')

        return {
            n: self.history[n]
            for n in names
        }

    def get_current(self):
        return {
            'state': self.state,
            'neighbors': self.neighbors,
            'neighbors_mask': self.neighbors_mask,
            'ci_ai_map': self.ci_ai_map,
            'metrics': self.metrics,
            **({} if self.secretes is None else {'secretes': self.secretes}),
        }

    # untested
    def write_history(self):
        if self.out_dir is not None:
            filename = os.path.join(self.out_dir, f"{self.episode + 1}.pt")
            chi = self.current_hist_idx + 1
            th.save(
                {
                    'agent_types': self.agent_types,
                    'actions': [string.ascii_uppercase[i] for i in range(self.n_actions)],
                    'agents': [string.ascii_uppercase[i] for i in range(self.n_nodes)],
                    'state': self.history['state'][:, :, :chi],
                    'reward': self.history['reward'][:, :, :chi],
                    'neighbors': self.history['neighbors'][:, :chi],
                    'neighbors_mask': self.history['neighbors_mask'][:, :chi],
                    'ci_ai_map': self.history['ci_ai_map'][:, :chi],
                    'episode': self.history['episode'][:chi],
                    'metrics': self.history['metrics'][:, :chi],
                    'metric_names': self.metric_names,
                    **({} if 'secretes' not in self.history else {'secretes': self.history['secretes'][:, :chi]}),
                },
                filename
            )

    # untested
    def finish_episode(self):
        if self.current_hist_idx == self.max_history - 1:
            self.write_history()

    # untested
    def to_dict(self):
        return self.info

    # TODO: unit test, merge with batch
    def _observe_neighbors(self, agent_type):
        current = self.get_current()
        current = trf.add_dimensions_to_current(**current)
        _current = trf.neighbors_states(**current, **self.info2)
        _current = trf.map_ci_agents(
            **_current, agent_type=agent_type)
        neighbors_states = _current['neighbors_states'].squeeze(1).squeeze(1)
        neighbors_mask = _current['neighbors_states_mask'].squeeze(1).squeeze(1)
        return neighbors_states, neighbors_mask

    # TODO: unit test
    def _batch_neighbors(self, episode_idx, agent_type):
        history = self.get_history()
        selected_history = trf.select_from_history(
            **history, episode_idx=episode_idx)
        _selected_history = trf.neighbors_states(**selected_history, **self.info2)
        _selected_history = trf.map_ci_agents(
            **_selected_history, agent_type=agent_type)

        return _selected_history

    def get_observation_shapes(self, agent_type):
        return {
            'neighbors_with_mask': ((self.n_nodes, self.max_neighbors + 1, 2), (self.n_nodes, self.max_neighbors + 1)),
            'neighbors': (self.n_nodes, self.max_neighbors + 1, 2),
            'matrix': ((self.n_nodes,), (self.n_nodes, self.max_neighbors, 2)),
            'neighbors_mask_secret_envinfo': (
                {'shape': (self.n_nodes, self.max_neighbors + 1, 2), 'maxval': self.n_actions},
                {'shape': (self.n_nodes, self.max_neighbors + 1), 'maxval': 1},
                get_secrets_shape(n_nodes=self.n_nodes,
                                  agent_type=agent_type, **self.secrete_args),
                {'shape': (self.n_nodes, len(self.metric_names)),
                 'maxval': 1, 'names': self.metric_names},
            ),
        }

    # TODO: End-to-End test
    def observe(self, mode, **kwarg):
        if mode == 'neighbors_with_mask':
            return self._observe_neighbors(**kwarg)
        elif mode == 'neighbors_mask_secret_envinfo':
            assert (kwarg['agent_type'] == 'ci') or (
                self.mapping_period <= 0), 'Obs mode does not allow agent mapping'
            obs, mask = self._observe_neighbors(**kwarg)
            if show_agent_type_secrets(**self.secrete_args, **kwarg):
                secretes = self.secretes
            else:
                secretes = None
            return obs, mask, secretes, self.metrics
        elif mode == 'neighbors':
            return self._observe_neighbors(**kwarg)[0]
        else:
            raise NotImplementedError(f'Mode {mode} is not implemented.')

    # TODO: End-to-End test
    def sample(self, agent_type, mode, batch_size, horizon=None, last=False, **kwarg):
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

        episode_idx = th.tensor([self.history_queue[pidx] for pidx in pos_idx], dtype=th.int64)

        _selected_history = self._batch_neighbors(
            episode_idx=episode_idx, agent_type=agent_type, **kwarg)

        modes_names = {
            'neighbors_mask_secret_envinfo': ['neighbors_states', 'neighbors_states_mask', 'secretes', 'metrics'],
            'neighbors_with_mask': ['neighbors_states', 'neighbors_states_mask']
        }

        # TODO: should be simplified
        block = [] if agent_type in self.secrete_args.get('agent_types', []) else ['secretes']
        previous, current = trf.shift_obs(
            _selected_history, modes_names[mode], block)

        actions = _selected_history['state'][self.agent_types_idx[agent_type], :, :, 1:]
        rewards = _selected_history['reward'][self.agent_types_idx[agent_type]]

        return previous, current, actions, rewards

    # TODO: Unit test.

    def _get_rewards2(self):
        current = self.get_current()
        current = trf.add_dimensions_to_current(**current)
        _current = trf.neighbors_states(**current, **self.info2)
        metrics = trf.calc_metrics(**_current)

        metrics = metrics.squeeze(1).squeeze(1)

        metrics_names = [f'{agg}_{m}' for agg in ['ind', 'local', 'avg']
                         for m in ['catch', 'coordination']]
        metrics_agent_idx = [j for i in range(3) for j in [1, 0]]

        ci_vec = th.tensor([self.rewards_args['ci'].get(m, 0)
                            for m in metrics_names], dtype=th.float)
        ai_vec = th.tensor([self.rewards_args['ai'].get(m, 0)
                            for m in metrics_names], dtype=th.float)

        ci_reward = metrics[:, metrics_agent_idx, range(6)] @ ci_vec
        ai_reward = metrics[:, metrics_agent_idx, range(6)] @ ai_vec

        m_idx = [metrics_names.index(mn) for mn in self.metric_names[:-1]]

        metrics = metrics[:, self.agent_types_idx['ci'], m_idx]

        metrics = th.cat([metrics, th.ones((*metrics.shape[:-1], 1))], dim=-1)

        return ci_reward, ai_reward, metrics

    def _get_rewards(self):
        ci_state = self.state[self.agent_types_idx['ci']]
        ai_state = self.state[self.agent_types_idx['ai']]

        ci_state_shifted = (ci_state + 1)
        ci_state_shifted_mapped = self.adjacency_matrix * \
            (ci_state + 1)  # [a,:] neighbors of agent a

        conflicts = (ci_state_shifted[:, np.newaxis] == ci_state_shifted_mapped)

        ind_coordination = 1 - conflicts.any(dim=0).type(th.float)
        ind_catch = (ai_state == ci_state).type(th.float)

        ad_matrix_float = self.adjacency_matrix.type(th.float)
        ad_matrix_float.fill_diagonal_(1.)
        ad_matrix_float = ad_matrix_float / ad_matrix_float.sum(0)
        local_coordination = ind_coordination @ ad_matrix_float
        local_catch = ind_catch @ ad_matrix_float
        # assert (local_catch <= 1).all()
        # assert (local_coordination <= 1).all()

        metrics = {
            'ind_coordination': ind_coordination,
            'avg_coordination': ind_coordination.mean(0, keepdim=True).expand(self.n_nodes),
            'local_coordination': local_coordination,
            'local_catch': local_catch,
            'ind_catch': ind_catch,
            'avg_catch': ind_catch.mean(0, keepdim=True).expand(self.n_nodes),
            'rewarded': th.full(
                size=(self.n_nodes,),
                fill_value=(self.episode_step % self.reward_period) == 0,
                dtype=th.float, device=self.device
            ),
        }

        ci_reward = th.stack(
            [metrics[k] * v for k, v in self.rewards_args['ci'].items()]).sum(0)*metrics['rewarded']
        ai_reward = th.stack(
            [metrics[k] * v for k, v in self.rewards_args['ai'].items()]).sum(0)*metrics['rewarded']

        metrics_stacked = th.stack([metrics[k] for k in self.metric_names], dim=1)

        return ci_reward, ai_reward, metrics_stacked

    # TODO: End-to-End test

    def step(self, actions):
        self.episode_step += 1

        assert actions['ci'].max() <= self.n_actions - 1
        assert actions['ai'].max() <= self.n_actions - 1
        assert actions['ai'].dtype == th.int64
        assert actions['ci'].dtype == th.int64

        if (self.episode_step + 1 == self.episode_steps):
            done = True
        elif self.episode_step >= self.episode_steps:
            raise ValueError('Environment is done already.')
        else:
            done = False

        self.state[self.agent_types_idx['ci']] = actions['ci']
        self.state[self.agent_types_idx['ai']] = actions['ai'][self.ai_ci_map]

        ci_reward, ai_reward, metrics = self._get_rewards()
        ci_reward2, ai_reward2, metrics2 = self._get_rewards2()

        assert ((metrics - metrics2).abs() < 0.000001) .all()
        assert (ci_reward == ci_reward2).all()
        assert (ai_reward == ai_reward2).all()

        self.metrics = metrics

        self._update_secrets()

        self.history['state'][:, :, self.current_hist_idx, self.episode_step + 1] = self.state
        self.history['reward'][self.agent_types_idx['ci'], :,
                               self.current_hist_idx, self.episode_step] = ci_reward
        self.history['reward'][self.agent_types_idx['ai'], :,
                               self.current_hist_idx, self.episode_step] = ai_reward
        self.history['metrics'][:, self.current_hist_idx, self.episode_step + 1] = self.metrics
        if 'secretes' in self.history:
            self.history['secretes'][:, self.current_hist_idx,
                                     self.episode_step + 1] = self.secretes

        rewards = {
            'ai': ai_reward[self.ci_ai_map],
            'ci': ci_reward
        }
        return rewards, done, {}

    def __del__(self):
        if self.current_hist_idx != self.max_history - 1:
            self.write_history()
