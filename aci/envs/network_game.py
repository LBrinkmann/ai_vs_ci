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
        return th.randint(low=0, high=n_seeds, size=(1,), device=device).expand(n_nodes)
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


def global_aggregation(metric):
    # metric: m, p, b, s
    n_metrics, n_nodes, batch_size, episode_steps = metric.shape
    return metric.mean(1, keepdim=True).expand(-1, n_nodes, -1, -1)


def local_aggregation(metric, neighbors, neighbors_mask):
    # metric: m, p, b, s
    # neighbors: p, b, n
    # neighbors_mask: p, b, n
    n_metrics, n_nodes, batch_size, episode_steps = metric.shape
    n_nodes_2, batch_size_2, max_neighbors = neighbors.shape

    assert n_nodes == n_nodes_2
    assert n_nodes == n_nodes_2

    # permutation needed because we map neighbors on agents
    _metric = metric.permute(0, 2, 3, 1) \
        .unsqueeze(1).expand(-1, n_nodes, -1, -1, -1)  # p, b, s, n

    _neighbors = neighbors.clone()
    _neighbors[neighbors_mask] = 0
    _neighbors = _neighbors.unsqueeze(0).unsqueeze(3).expand(
        n_metrics, -1, -1, episode_steps, 1)  # p, b, s, n

    _neighbors_mask = neighbors_mask.unsqueeze(0).unsqueeze(3) \
        .expand(n_metrics, -1, -1, episode_steps, 1)  # p, b, s, n

    local_metrics = th.gather(_metric, -1, _neighbors)  # t, p, b, s, n
    local_metrics[_neighbors_mask] = 0
    local_metrics = local_metrics.sum(-1) / (~_neighbors_mask).sum(-1)
    return local_metrics


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

    # TODO: understand and document
    def _update_secrets(self):
        if do_update_secrets(episode_step=self.episode_step, **self.secrete_args):
            self.secretes = get_secrets(
                **self.secrete_args, n_nodes=self.n_nodes, device=self.device)

    # untested
    def init_history(self):
        # shape0 = (self.max_history,)
        # shape1 = (len(self.agent_types), self.n_nodes, self.max_history, self.episode_steps + 1)
        # shape2 = (len(self.agent_types), self.n_nodes, self.max_history, self.episode_steps)
        # shape3 = (self.n_nodes, self.max_history, self.max_neighbors + 1)
        # shape4 = (self.n_nodes, self.max_history)
        # shape5 = (self.n_nodes, self.max_history, self.episode_steps + 1, len(self.metric_names))
        # shape6 = (self.n_nodes, self.max_history, self.episode_steps + 1)

        # newshape0 = (self.max_history,)  # h
        # newshape1 = (self.max_history, self.episode_steps + 1, self.n_nodes,
        #              len(self.agent_types))  # h e n a .permute(3,2,0,1)
        # newshape2 = (self.max_history, self.episode_steps, self.n_nodes,
        #              len(self.agent_types), )  # .permute(3,2,0,1)
        # newshape3 = (self.max_history, self.n_nodes, self.max_neighbors + 1)  # .permute(1,0,2)
        # newshape4 = (self.max_history, self.n_nodes)  # .permute(1,0)
        # newshape5 = (self.max_history, self.episode_steps + 1, self.n_nodes,
        #              len(self.metric_names))  # .permute(2,0,1,3)
        # newshape6 = (self.max_history, self.episode_steps + 1, self.n_nodes)  # .permute(2,0,1)

        self.history_queue = collections.deque([], maxlen=self.max_history)
        self.episode_history = th.empty(
            (self.max_history,), dtype=th.int64, device=self.device)  # h
        self.state_history = th.empty((self.max_history, self.episode_steps + 1, self.n_nodes,
                                       len(self.agent_types)), dtype=th.int64, device=self.device)  # h s+ p t
        self.reward_history = th.empty((self.max_history, self.episode_steps, self.n_nodes,
                                        len(self.agent_types), ), dtype=th.float32, device=self.device)  # h s p t

        self.neighbors_history = th.empty(
            (self.max_history, self.n_nodes, self.max_neighbors + 1), dtype=th.int64, device=self.device)  # h p n+
        self.neighbors_mask_history = th.empty(
            (self.max_history, self.n_nodes, self.max_neighbors + 1), dtype=th.bool, device=self.device)  # h p n+
        self.ci_ai_map_history = th.empty(
            (self.max_history, self.n_nodes), dtype=th.int64, device=self.device)  # h p

        self.metrics_history = th.empty((self.max_history, self.episode_steps + 1, self.n_nodes,
                                         len(self.metric_names)), dtype=th.float, device=self.device)  # h s+ p m

        if gen_secrets(**self.secrete_args):
            self.secretes_history = th.empty(
                (self.max_history, self.episode_steps + 1, self.n_nodes), dtype=th.int64, device=self.device)  # h s+ p
        else:
            self.secretes_history = None

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
        self.episode_history[self.current_hist_idx] = self.episode
        self.state_history[self.current_hist_idx, self.episode_step + 1] = self.state.T
        self.neighbors_history[self.current_hist_idx] = self.neighbors
        self.neighbors_mask_history[self.current_hist_idx] = self.neighbors_mask

        if self.secretes_history is not None:
            self.secretes_history[self.current_hist_idx, self.episode_step + 1] = self.secretes
        self.metrics_history[self.current_hist_idx, self.episode_step + 1] = self.metrics

        self.ci_ai_map_history[self.current_hist_idx] = self.ci_ai_map

        self.info = {
            'graph': self.graph,
            'graph_info': self.graph_info,
            'ai_ci_map': self.ai_ci_map.tolist(),
            'ci_ai_map': self.ci_ai_map.tolist()
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
                    'state': self.state_history.permute(3, 2, 0, 1)[:, :, :chi],
                    'reward': self.reward_history.permute(3, 2, 0, 1)[:, :, :chi],
                    'neighbors': self.neighbors_history.permute(1, 0, 2)[:, :chi],
                    'neighbors_mask': self.neighbors_mask_history.permute(1, 0, 2)[:, :chi],
                    'ci_ai_map': self.ci_ai_map_history.permute(1, 0)[:, :chi],
                    'episode': self.episode_history[:chi],
                    'metrics': self.metrics_history.permute(2, 0, 1, 3)[:, :chi],
                    'metric_names': self.metric_names,
                    **({} if self.secretes_history is None else {'secretes': self.secretes_history.permute(2, 0, 1)[:, :chi]}),
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
        _state = self.state.unsqueeze(-1).unsqueeze(-1)  # t p * *
        _neighbors = self.neighbors.unsqueeze(1)  # p * n+
        _neighbors_mask = self.neighbors_mask.unsqueeze(1)  # p * n+

        observations = self.get_observations(_state, _neighbors, _neighbors_mask)  # t p * * n+

        observations = observations.permute(1, 4, 0, 2, 3).squeeze(-1).squeeze(-1)  # p n+ t

        if agent_type == 'ci':
            return observations, self.neighbors_mask
        elif agent_type == 'ai':
            return observations[self.ci_ai_map], self.neighbors_mask[self.ci_ai_map]
        else:
            raise ValueError(f'Unkown agent_type {agent_type}')

    # TODO: unit test
    def _batch_neighbors(self, episode_idx, agent_type):
        _states = self.state_history.permute(3, 2, 0, 1)[:, :, episode_idx]  # h s+ p t => t p b s+
        _neighbors = self.neighbors_history.permute(1, 0, 2)[:, episode_idx]  # h p n+ => p b n+
        _neighbors_mask = self.neighbors_mask_history.permute(
            1, 0, 2)[:, episode_idx]  # h p n+ => p b n+

        observations = self.get_observations(_states, _neighbors, _neighbors_mask)  # t p b s+ n+

        _neighbors_mask = _neighbors_mask.unsqueeze(-2) \
            .expand(-1, -1, self.episode_steps + 1, -1)  # p b s+ n+
        obs = observations.permute(1, 2, 3, 4, 0)  # p b s+ n+ t
        if agent_type == 'ci':
            pass
        elif agent_type == 'ai':
            _ci_ai_map_history = self.ci_ai_map_history.permute(1, 0)[:, episode_idx] \
                .unsqueeze(-1) \
                .unsqueeze(-1) \
                .unsqueeze(-1) \
                .expand(-1, -1, obs.shape[2], obs.shape[3], obs.shape[4])  # p b s+ n+ t
            obs = th.gather(obs, 0, _ci_ai_map_history)  # p b s+ n+ t
            _neighbors_mask = th.gather(
                _neighbors_mask, 0, _ci_ai_map_history[:, :, :, :, 0])  # p b s+ n+
        else:
            raise ValueError(f'Unkown agent_type {agent_type}')

        prev_observations = obs[:, :, :-1]  # p b s n+ t
        observations = obs[:, :, 1:]  # p b s n+ t
        prev_mask = _neighbors_mask[:, :, :-1]  # p b s n+
        mask = _neighbors_mask[:, :, 1:]  # p b s n+

        assert prev_observations.max() <= self.n_actions - 1
        assert observations.max() <= self.n_actions - 1

        return (prev_observations, prev_mask), (observations, mask)

    def _observe_matrix(self, agent_type):
        raise NotImplementedError('Not tested, currently not supported')
        ci_state = self.state[self.agent_types_idx['ci']]

        if (agent_type == 'ai') and (self.mapping_period > 0):
            raise NotImplementedError("Matrix observation only implemented with fixed mapping")

        links = th.stack([
            self.neighbors[:, [0]].expand(-1, self.neighbors.shape[-1] - 1),
            self.neighbors[:, 1:]
        ], dim=-1)

        return ci_state, links, self.neighbors_mask[:, 1:]

    def _batch_matrix(self, episode_idx, agent_type):
        raise NotImplementedError('Not tested, currently not supported')
        if (agent_type == 'ai') and (self.mapping_period > 0):
            raise NotImplementedError("Matrix observation only implemented with fixed mapping")

        ci_state = self.state_history.permute(
            3, 2, 0, 1)[self.agent_types_idx['ci'], :, episode_idx]
        mask = self.neighbors_mask_history.permute(1, 0, 2)[:, episode_idx]
        neighbors = self.neighbors_history.permute(1, 0, 2)[:, episode_idx]
        links = th.stack([
            neighbors[:, :, [0]].expand(-1, -1, neighbors.shape[-1] - 1),
            neighbors[:, :, 1:]
        ], dim=-1)
        links = links.unsqueeze(2).expand(-1, -1, self.episode_steps, -1, -1)
        mask = mask[:, :, 1:].unsqueeze(2).expand(-1, -1, self.episode_steps, -1)
        prev_states = ci_state[:, :, :-1]
        this_states = ci_state[:, :, 1:]
        return (prev_states, links, mask), (this_states, links, mask)

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
        elif mode == 'matrix':
            return self._observe_matrix(**kwarg)
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

        if mode in ['neighbors_with_mask', 'neighbors', 'neighbors_mask_secret_envinfo']:
            prev_obs, obs = self._batch_neighbors(
                episode_idx=episode_idx, agent_type=agent_type, **kwarg)
            if mode == 'neighbors':
                prev_obs = prev_obs[0]
                obs = obs[0]
            elif mode == 'neighbors_mask_secret_envinfo':
                if self.metrics_history is not None:
                    all_env_state = self.metrics_history.permute(2, 0, 1, 3)[:, episode_idx]
                    prev_env_state = all_env_state[:, :, :-1]
                    env_state = all_env_state[:, :, 1:]
                else:
                    prev_env_state, env_state = None, None
                if show_agent_type_secrets(agent_type=agent_type, **self.secrete_args):
                    all_secrets = self.secretes_history.permute(2, 0, 1)[:, episode_idx]
                    prev_secrets = all_secrets[:, :, :-1]
                    secrets = all_secrets[:, :, 1:]
                else:
                    prev_secrets = None
                    secrets = None
                obs = (*obs, secrets, env_state)
                prev_obs = (*prev_obs, prev_secrets, prev_env_state)
        elif mode == 'matrix':
            prev_obs, obs = self._batch_matrix(
                episode_idx=episode_idx, agent_type=agent_type, **kwarg)
        else:
            raise NotImplementedError(f'Mode {mode} is not implemented.')

        actions = self.state_history.permute(
            3, 2, 0, 1)[self.agent_types_idx[agent_type], :, episode_idx, 1:]
        rewards = self.reward_history.permute(
            3, 2, 0, 1)[self.agent_types_idx[agent_type], :, episode_idx]
        if agent_type == 'ai':
            _ci_ai_map_history = self.ci_ai_map_history.permute(1, 0)[:, episode_idx] \
                .unsqueeze(-1) \
                .expand(-1, -1, self.episode_steps)
            actions = th.gather(actions, 0, _ci_ai_map_history)
            rewards = th.gather(rewards, 0, _ci_ai_map_history)

        assert actions.max() <= self.n_actions - 1
        return prev_obs, obs, actions, rewards

    # TODO: End-to-End test
    def _map_incoming(self, values):
        return {'ai': values['ai'][self.ai_ci_map], 'ci': values['ci']}

    # TODO: End-to-End test
    def _map_outgoing(self, values):
        return {'ai': values['ai'][self.ci_ai_map], 'ci': values['ci']}

    # new api
    @staticmethod
    def get_observations(state, neighbors, neighbors_mask, **_):
        # state: t, p, b, s
        # neighbors: p, b, n
        # neighbors_mask: p, b, n
        o_observations = o_get_observations(
            state, neighbors, neighbors_mask, **_)

        _state = state.permute(2, 3, 1, 0)  # h s+ p t
        _neighbors = neighbors.permute(1, 0, 2)  # h p n+
        _neighbors_mask = neighbors_mask.permute(1, 0, 2)  # h p n+
        n_observations = NetworkGame.n_get_observations(
            _state, _neighbors, _neighbors_mask)  # b s p n+ t
        n_observations = n_observations.permute(4, 2, 0, 1, 3)  # t, p, b, s, n
        assert (o_observations == n_observations).all()
        return n_observations

    # new api
    @staticmethod
    def n_get_observations(state, neighbors, neighbors_mask, **_):
        # state: t, p, b, s ==> b s p t
        # neighbors: p, b, n ==> b p n+
        # neighbors_mask: p, b, n ==> b p n+
        batch_size, episode_steps, n_nodes, n_agent_types = state.shape
        batch_size_2, n_nodes_2, max_neighbors = neighbors.shape
        assert n_nodes == n_nodes_2
        assert batch_size == batch_size_2

        # agent position on neighbor position
        neighbor_state = state.unsqueeze(-2).expand(-1, -1, -1, -1, n_nodes, -1)  # b s p t

        _neighbors = neighbors.clone()
        _neighbors[neighbors_mask] = 0
        _neighbors = _neighbors.unsqueeze(0).unsqueeze(3).expand(
            n_agent_types, -1, -1, episode_steps, -1)  # t, p, b, s, n

        observations = th.gather(_state, -1, _neighbors)  # b s p n t
        observations[_neighbors_mask] = -1  # b s p n t

        .permute(0, 2, 3, 1)  # t, b, s, p,
        _state = _state.unsqueeze(1).expand(-1, n_nodes, -1, -1, -1)  # t, p, b, s, n

        _state = state.permute(0, 2, 3, 1)  # t, p, b, s
        _state = _state.unsqueeze(1).expand(-1, n_nodes, -1, -1, -1)  # t, p, b, s, n

        _neighbors_mask = neighbors_mask \
            .unsqueeze(0).unsqueeze(3) \
            .expand(n_agent_types, -1, -1, episode_steps, -1)  # t, p, b, s, n

        observations = th.gather(_state, -1, _neighbors)  # t, p, b, s, n
        observations[_neighbors_mask] = -1  # t, p, b, s, n
        return observations  # t, p, b, s, n

    # old api
    @staticmethod
    def o_get_observations(state, neighbors, neighbors_mask, **_):
        # state: t, p, b, s
        # neighbors: p, b, n
        # neighbors_mask: p, b, n
        n_agent_types, n_nodes, batch_size, episode_steps = state.shape
        n_nodes_2, batch_size_2, max_neighbors = neighbors.shape

        assert n_nodes == n_nodes_2
        assert n_nodes == n_nodes_2

        # permutation needed because we map neighbors on agents
        _state = state.permute(0, 2, 3, 1) \
            .unsqueeze(1).expand(-1, n_nodes, -1, -1, -1)  # t, p, b, s, n

        _neighbors = neighbors.clone()
        _neighbors[neighbors_mask] = 0
        _neighbors = _neighbors.unsqueeze(0).unsqueeze(3).expand(
            n_agent_types, -1, -1, episode_steps, -1)  # t, p, b, s, n

        _neighbors_mask = neighbors_mask \
            .unsqueeze(0).unsqueeze(3) \
            .expand(n_agent_types, -1, -1, episode_steps, -1)  # t, p, b, s, n

        observations = th.gather(_state, -1, _neighbors)  # t, p, b, s, n
        observations[_neighbors_mask] = -1  # t, p, b, s, n
        return observations  # t, p, b, s, n

    @staticmethod
    def get_metrics(observations, neighbors, neighbors_mask, ci_idx, ai_idx):
        """
            Args:
                observations: b s p n+ t
                neighbors:

            Returns:
                b s p m
        """
        ci_color = observations[:, :, :, 0, ci_idx]  # b s p
        ai_color = observations[:, :, :, 0, ai_idx]  # b s p
        ci_neighbor_colors = observations[:, :, :, 1:, ci_idx]  # b s p n

        ci_anticoor = (ci_color.unsqueeze(-1) != ci_neighbor_colors) \
            .all(-1).type(th.float)  # b s p
        ci_coor = (ci_color.unsqueeze(-1) == ci_neighbor_colors) \
            .all(-1).type(th.float)  # b s p
        catch = (ai_color == ci_color).type(th.float)  # b s p
        ind_metrics = th.stack([ci_anticoor, ci_coor, catch], dim=-1)  # b s p m/3

        local_metrics = local_aggregation(ind_metrics, neighbors, neighbors_mask)  # b s p m/3
        global_metrics = global_aggregation(ind_metrics)  # b s p m/3

        metrics = th.stack([ind_metrics, local_metrics, global_metrics], dim=-1)  # b s p m

        return metrics  # b s p m

    # TODO: Unit test.

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

        _state = self.state.unsqueeze(-1).unsqueeze(-1)  # t p * *
        _neighbors = self.neighbors.unsqueeze(1)  # p * n+
        _neighbors_mask = self.neighbors_mask.unsqueeze(1)  # p * n+

        observations = self._get_observations(_state, _neighbors, _neighbors_mask)  # t, p, b, s, n

        _state = self.state.unsqueeze(-1).unsqueeze(-1)  # t p * *
        _neighbors = self.neighbors.unsqueeze(1)  # p * n+
        _neighbors_mask = self.neighbors_mask.unsqueeze(1)  # p * n+
        metrics_stacked2 = self.get_metrics(
            _observations, _neighbors, _neighbors_mask,
            self.agent_types_idx['ci'], self.agent_types_idx['ai'])

        import ipdb
        ipdb.set_trace()

        _metrics_stacked2 = metrics_stacked2[:, :, :, [0, 6, 3, 5, 2, 8]]  # b s p m

        assert _metrics_stacked2 == metrics_stacked[:, :, :, ]

        # self.metric_names = [
        #     'ind_coordination', 'avg_coordination', 'local_coordination', 'local_catch',
        #     'ind_catch', 'avg_catch', 'rewarded'
        # ]

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

        self.metrics = metrics

        self._update_secrets()

        self.state_history[self.current_hist_idx, self.episode_step + 1] = self.state.T
        self.reward_history[self.current_hist_idx, self.episode_step,
                            :, self.agent_types_idx['ci']] = ci_reward
        self.reward_history[self.current_hist_idx, self.episode_step,
                            :, self.agent_types_idx['ai']] = ai_reward
        self.metrics_history[self.current_hist_idx, self.episode_step + 1] = self.metrics
        if self.secretes_history is not None:
            self.secretes_history[self.current_hist_idx, self.episode_step + 1] = self.secretes

        rewards = {
            'ai': ai_reward[self.ci_ai_map],
            'ci': ci_reward
        }
        return rewards, done, {}

    def __del__(self):
        if self.current_hist_idx != self.max_history - 1:
            self.write_history()
