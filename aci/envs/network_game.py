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
        self.max_degree = determine_max_degree(n_nodes=n_nodes, **graph_args)
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
        self.history_queue = collections.deque([], maxlen=self.max_history)
        self.episode_history = th.empty((self.max_history,), dtype=th.int64, device=self.device)
        self.state_history = th.empty(
            (len(self.agent_types), self.n_nodes, self.max_history, self.episode_steps + 1), dtype=th.int64, device=self.device)
        self.reward_history = th.empty(
            (len(self.agent_types), self.n_nodes, self.max_history, self.episode_steps), dtype=th.float32, device=self.device)

        self.neighbors_history = th.empty(
            (self.n_nodes, self.max_history, self.max_degree + 1), dtype=th.int64, device=self.device)
        self.neighbors_mask_history = th.empty(
            (self.n_nodes, self.max_history, self.max_degree + 1), dtype=th.bool, device=self.device)
        self.ci_ai_map_history = th.empty(
            (self.n_nodes, self.max_history), dtype=th.int64, device=self.device)

        self.metrics_history = th.empty(
            (self.n_nodes, self.max_history, self.episode_steps + 1, len(self.metric_names)), dtype=th.float, device=self.device)

        if gen_secrets(**self.secrete_args):
            self.secretes_history = th.empty(
                (self.n_nodes, self.max_history, self.episode_steps + 1), dtype=th.int64, device=self.device)
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
            padded_neighbors, neighbors_mask = pad_neighbors(neighbors, self.max_degree)
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
        self.state_history[:, :, self.current_hist_idx, self.episode_step + 1] = self.state
        self.neighbors_history[:, self.current_hist_idx] = self.neighbors
        self.neighbors_mask_history[:, self.current_hist_idx] = self.neighbors_mask

        if self.secretes_history is not None:
            self.secretes_history[:, self.current_hist_idx, self.episode_step + 1] = self.secretes
        self.metrics_history[:, self.current_hist_idx, self.episode_step + 1] = self.metrics

        self.ci_ai_map_history[:, self.current_hist_idx] = self.ci_ai_map

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
                    'state': self.state_history[:, :, :chi],
                    'reward': self.reward_history[:, :, :chi],
                    'neighbors': self.neighbors_history[:, :chi],
                    'neighbors_mask': self.neighbors_mask_history[:, :chi],
                    'ci_ai_map': self.ci_ai_map_history[:, :chi],
                    'episode': self.episode_history[:chi],
                    'metrics': self.metrics_history[:, :chi],
                    'metric_names': self.metric_names,
                    **({} if self.secretes_history is None else {'secretes': self.secretes_history[:, :chi]}),
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
        state = self.state.T.unsqueeze(0).repeat(
            self.n_nodes, 1, 1)  # repeated, agent, agent_type

        _neighbors = self.neighbors.clone()
        _neighbors[self.neighbors_mask] = 0
        _neighbors = _neighbors.unsqueeze(-1).repeat(1, 1, 2)
        obs = th.gather(state, 1, _neighbors)
        obs[self.neighbors_mask] = -1

        if agent_type == 'ci':
            return obs, self.neighbors_mask
        elif agent_type == 'ai':
            return obs[self.ci_ai_map], self.neighbors_mask[self.ci_ai_map]
        else:
            raise ValueError(f'Unkown agent_type {agent_type}')

    # TODO: unit test
    def _batch_neighbors(self, episode_idx, agent_type):
        eps_states = self.state_history[:, :, episode_idx]

        eps_states = eps_states.permute(2, 3, 1, 0).unsqueeze(0).repeat(self.n_nodes, 1, 1, 1, 1)

        _mask = self.neighbors_mask_history[:, episode_idx]
        _neighbors = self.neighbors_history[:, episode_idx].clone()
        _neighbors[_mask] = 0
        _neighbors = _neighbors.unsqueeze(
            2).unsqueeze(-1).repeat(1, 1, self.episode_steps + 1, 1, 2)
        _mask = _mask.unsqueeze(-2).repeat(1, 1, self.episode_steps + 1, 1)

        obs = th.gather(eps_states, -2, _neighbors)
        obs[_mask] = -1
        if agent_type == 'ci':
            pass
        elif agent_type == 'ai':
            _ci_ai_map_history = self.ci_ai_map_history[:, episode_idx] \
                .unsqueeze(-1) \
                .unsqueeze(-1) \
                .unsqueeze(-1) \
                .repeat(1, 1, obs.shape[2], obs.shape[3], obs.shape[4])
            obs = th.gather(obs, 0, _ci_ai_map_history)
            _mask = th.gather(_mask, 0, _ci_ai_map_history[:, :, :, :, 0])
        else:
            raise ValueError(f'Unkown agent_type {agent_type}')

        prev_observations = obs[:, :, :-1]
        observations = obs[:, :, 1:]
        prev_mask = _mask[:, :, :-1]
        mask = _mask[:, :, 1:]

        assert prev_observations.max() <= self.n_actions - 1
        assert observations.max() <= self.n_actions - 1

        return (prev_observations, prev_mask), (observations, mask)

    def _observe_matrix(self, agent_type):
        ci_state = self.state[self.agent_types_idx['ci']]

        if (agent_type == 'ai') and (self.mapping_period > 0):
            raise NotImplementedError("Matrix observation only implemented with fixed mapping")

        links = th.stack([
            self.neighbors[:, [0]].repeat(1, self.neighbors.shape[-1] - 1),
            self.neighbors[:, 1:]
        ], dim=-1)

        return ci_state, links, self.neighbors_mask[:, 1:]

    def _batch_matrix(self, episode_idx, agent_type):
        if (agent_type == 'ai') and (self.mapping_period > 0):
            raise NotImplementedError("Matrix observation only implemented with fixed mapping")

        ci_state = self.state_history[self.agent_types_idx['ci'], :, episode_idx]
        mask = self.neighbors_mask_history[:, episode_idx]
        neighbors = self.neighbors_history[:, episode_idx]
        links = th.stack([
            neighbors[:, :, [0]].repeat(1, 1, neighbors.shape[-1] - 1),
            neighbors[:, :, 1:]
        ], dim=-1)
        links = links.unsqueeze(2).repeat(1, 1, self.episode_steps, 1, 1)
        mask = mask[:, :, 1:].unsqueeze(2).repeat(1, 1, self.episode_steps, 1)
        prev_states = ci_state[:, :, :-1]
        this_states = ci_state[:, :, 1:]
        return (prev_states, links, mask), (this_states, links, mask)

    def get_observation_shapes(self, agent_type):
        return {
            'neighbors_with_mask': ((self.n_nodes, self.max_degree + 1, 2), (self.n_nodes, self.max_degree + 1)),
            'neighbors': (self.n_nodes, self.max_degree + 1, 2),
            'matrix': ((self.n_nodes,), (self.n_nodes, self.max_degree, 2)),
            'neighbors_mask_secret_envinfo': (
                {'shape': (self.n_nodes, self.max_degree + 1, 2), 'maxval': self.n_actions},
                {'shape': (self.n_nodes, self.max_degree + 1), 'maxval': 1},
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
                    all_env_state = self.metrics_history[:, episode_idx]
                    prev_env_state = all_env_state[:, :, :-1]
                    env_state = all_env_state[:, :, 1:]
                else:
                    prev_env_state, env_state = None, None
                if show_agent_type_secrets(agent_type=agent_type, **self.secrete_args):
                    all_secrets = self.secretes_history[:, episode_idx]
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

        actions = self.state_history[self.agent_types_idx[agent_type], :, episode_idx, 1:]
        rewards = self.reward_history[self.agent_types_idx[agent_type], :, episode_idx]
        if agent_type == 'ai':
            _ci_ai_map_history = self.ci_ai_map_history[:, episode_idx] \
                .unsqueeze(-1) \
                .repeat(1, 1, self.episode_steps)
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

        self.state_history[:, :, self.current_hist_idx, self.episode_step + 1] = self.state
        self.reward_history[self.agent_types_idx['ci'], :,
                            self.current_hist_idx, self.episode_step] = ci_reward
        self.reward_history[self.agent_types_idx['ai'], :,
                            self.current_hist_idx, self.episode_step] = ai_reward
        self.metrics_history[:, self.current_hist_idx, self.episode_step + 1] = self.metrics
        if self.secretes_history is not None:
            self.secretes_history[:, self.current_hist_idx, self.episode_step + 1] = self.secretes

        rewards = {
            'ai': ai_reward[self.ci_ai_map],
            'ci': ci_reward
        }
        return rewards, done, {}

    def __del__(self):
        if self.current_hist_idx != self.max_history - 1:
            self.write_history()
