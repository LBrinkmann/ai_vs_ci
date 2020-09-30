"""
Graph Coloring Environment
"""

import torch as th
import numpy as np
import collections
from .utils.graph import create_graph, determine_max_degree, pad_neighbors


def show_agent_type_secrets(agent_type=None, agent_types=None, n_seeds=None, **kwargs):
    if (n_seeds is not None) and (agent_type is not None) and (agent_type in agent_types):
        return True
    else:
        return False

def get_secrets(n_agents=None, n_seeds=None, correlated=None, device=None, **kwargs):
    if n_seeds is None:
        return None
    assert (n_agents is not None)
    assert (correlated is not None)
    if correlated:
        return th.randint(low=1, high=n_seeds, size=(1,), device=device).repeat(n_agents)
    else:
        return th.randint(low=1, high=n_seeds, size=(n_agents,), device=device)


def get_secrets_shape(n_agents, n_seeds=None, agent_type=None, agent_types=None, **kwargs):
    if (n_seeds is not None) and (agent_type in agent_types):
        return {'shape': (n_agents,), 'maxval': n_seeds - 1}
    else:
        return None


def do_update_secrets(secret_period, episode_step, **kwars):
    return (episode_step % secret_period == 0) or (episode_step == 0)


def gen_secrets(n_seeds=None, **kwargs):
    if (n_seeds is not None):
        return True
    else:
        return False


def get_envinfo(info, info_names=[]):
    if len(info_names) == 0:
        return None
    else:
        return th.stack([info[name] for name in info_names], dim=1)


def get_envinfo_shape(n_agents, info_names=[]):
    if len(info_names) == 0:
        return None
    else:
        return {'shape': (n_agents, len(info_names),), 'maxval': 1}


class AdGraphColoringHist():
    def __init__(
            self, n_agents, n_actions, graph_args, episode_length, max_history, mapping_period, 
            network_period, rewards_args, device, reward_period, secrete_args={}, envinfo_args={}):
        self.episode_length = episode_length
        self.mapping_period = mapping_period
        self.network_period = network_period
        self.reward_period = reward_period
        self.max_history = max_history
        self.graph_args = graph_args
        self.rewards_args = rewards_args
        self.secrete_args = secrete_args
        self.envinfo_args = envinfo_args
        self.agent_types = ['ci', 'ai']
        self.agent_types_idx = {'ci': 0, 'ai': 1}
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.episode = -1
        self.max_degree = determine_max_degree(n_nodes=n_agents, **graph_args)

        self.device = device
        self.init_history()

    def _update_secrets(self):
        if do_update_secrets(episode_step=self.episode_step, **self.secrete_args):
            self.secretes = get_secrets(**self.secrete_args, n_agents=self.n_agents, device=self.device)

    def _get_envinfo(self, info):
        return get_envinfo(info=info, **self.envinfo_args)

    def init_history(self):
        self.history_queue = collections.deque([], maxlen=self.max_history)
        self.state_history = th.empty(
            (len(self.agent_types), self.n_agents, self.max_history, self.episode_length + 1), dtype=th.int64, device=self.device)
        self.reward_history = th.empty(
            (len(self.agent_types), self.n_agents, self.max_history, self.episode_length), dtype=th.float32, device=self.device)

        self.neighbors_history = th.empty((self.n_agents, self.max_history, self.max_degree + 1), dtype=th.int64, device=self.device)
        self.neighbors_mask_history = th.empty((self.n_agents, self.max_history, self.max_degree + 1), dtype=th.bool, device=self.device)
        self.ci_ai_map_history = th.empty((self.n_agents, self.max_history), dtype=th.int64, device=self.device)

        envinfo_shape = get_envinfo_shape(self.n_agents, **self.envinfo_args)
        if envinfo_shape is not None:
            self.envinfo_history = th.empty(
                (self.n_agents, self.max_history, self.episode_length + 1, envinfo_shape['shape'][1]), dtype=th.float, device=self.device)
        else:
            self.envinfo_history = None

        if gen_secrets(**self.secrete_args):
            self.secretes_history = th.empty((self.n_agents, self.max_history, self.episode_length + 1), dtype=th.int64, device=self.device)
        else:
            self.secretes_history = None

    def init_episode(self):
        self.episode += 1
        self.episode_step = -1
        self.current_hist_idx = self.episode % self.max_history

        # create new graph (either only on first episode, or every network_period episode)
        if (self.episode == 0) or (
            (self.network_period > 0) and (self.episode % self.network_period == 0)):
            self.graph, neighbors, adjacency_matrix, self.graph_info = create_graph(
                n_nodes=self.n_agents, **self.graph_args)
            padded_neighbors, neighbors_mask = pad_neighbors(neighbors, self.max_degree)
            self.neighbors = th.tensor(padded_neighbors, dtype=th.int64, device=self.device)
            self.neighbors_mask = th.tensor(neighbors_mask, dtype=th.bool, device=self.device)
            self.adjacency_matrix = th.tensor(
                adjacency_matrix, dtype=th.bool, device=self.device)
            
            
        # mapping between ai agents and ci agents (outward: ci_ai_map, incoming: ai_ci_map)
        # ci_ai_map: [2, 0, 1], ai agent 0 is ci agent 2
        if (self.mapping_period > 0) and (self.episode % self.mapping_period == 0):
            self.ai_ci_map = th.randperm(self.n_agents)
            self.ci_ai_map = th.argsort(self.ai_ci_map)
        elif self.episode == 0:
            self.ai_ci_map = th.arange(self.n_agents)
            self.ci_ai_map = th.argsort(self.ai_ci_map)

        # init state
        self.state = th.tensor(
                np.random.randint(0, self.n_actions, (2, self.n_agents)), dtype=th.int64, device=self.device)
        self._update_secrets()

        ci_reward, ai_reward, info = self._get_rewards()
        self.envinfo = self._get_envinfo(info=info)

        # add to history
        self.history_queue.appendleft(self.current_hist_idx)
        self.state_history[:, :, self.current_hist_idx, self.episode_step + 1] = self.state
        # self.adjacency_matrix_history[self.current_hist_idx] = self.adjacency_matrix
        self.neighbors_history[:, self.current_hist_idx] = self.neighbors
        self.neighbors_mask_history[:, self.current_hist_idx] = self.neighbors_mask

        if self.secretes_history is not None:
            self.secretes_history[:, self.current_hist_idx, self.episode_step + 1] = self.secretes
        if self.envinfo_history is not None:
            self.envinfo_history[:, self.current_hist_idx, self.episode_step + 1] = self.envinfo

        self.ci_ai_map_history[:, self.current_hist_idx] = self.ci_ai_map

        self.info = {
            'graph': self.graph,
            'graph_info': self.graph_info,
            'ai_ci_map': self.ai_ci_map.tolist(),
            'ci_ai_map': self.ci_ai_map.tolist()
        }

    def to_dict(self):
        return self.info

    def _observe_neighbors(self, agent_type):
        state = self.state.T.unsqueeze(0).repeat(self.n_agents,1,1) # repeated, agent, agent_type

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

    
    def _batch_neighbors(self, episode_idx, agent_type):
        eps_states = self.state_history[:, :, episode_idx]

        eps_states = eps_states.permute(2,3,1,0).unsqueeze(0).repeat(self.n_agents,1,1,1,1)

        _mask = self.neighbors_mask_history[:, episode_idx]
        _neighbors = self.neighbors_history[:, episode_idx].clone()
        _neighbors[_mask] = 0
        _neighbors = _neighbors.unsqueeze(2).unsqueeze(-1).repeat(1,1,self.episode_length + 1,1,2)
        _mask = _mask.unsqueeze(-2).repeat(1,1,self.episode_length + 1,1)

        obs = th.gather(eps_states, -2, _neighbors)
        obs[_mask] = -1
        if agent_type == 'ci':
            pass
        elif agent_type == 'ai':
            _ci_ai_map_history = self.ci_ai_map_history[:, episode_idx] \
                .unsqueeze(-1) \
                .unsqueeze(-1) \
                .unsqueeze(-1) \
                .repeat(1,1,obs.shape[2],obs.shape[3],obs.shape[4])
            obs = th.gather(obs, 0, _ci_ai_map_history)
            _mask = th.gather(_mask, 0, _ci_ai_map_history[:,:,:,:,0])
        else:
            raise ValueError(f'Unkown agent_type {agent_type}')

        prev_observations = obs[:,:,:-1]
        observations = obs[:,:,1:]
        prev_mask = _mask[:,:,:-1]
        mask = _mask[:,:,1:]

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

        links = links.unsqueeze(2).repeat(1,1,self.episode_length,1,1)

        mask = mask[:, :, 1:].unsqueeze(2).repeat(1,1,self.episode_length,1)
        
        prev_states = ci_state[:,:,:-1]
        this_states = ci_state[:,:,1:]

        return (prev_states, links, mask), (this_states, links, mask) 


    def get_observation_shapes(self, agent_type):
        return {
            'neighbors_with_mask': ((self.n_agents, self.max_degree + 1, 2), (self.n_agents, self.max_degree + 1)),
            'neighbors': (self.n_agents, self.max_degree + 1, 2),
            'matrix': ((self.n_agents,), (self.n_agents, self.max_degree, 2)),
            'neighbors_mask_secret_envinfo': (
                {'shape': (self.n_agents, self.max_degree + 1, 2), 'maxval': self.n_actions},
                {'shape': (self.n_agents, self.max_degree + 1), 'maxval': 1},
                get_secrets_shape(n_agents=self.n_agents, agent_type=agent_type, **self.secrete_args),
                get_envinfo_shape(n_agents=self.n_agents, **self.envinfo_args),
            ),
        }

    def observe(self, mode, **kwarg):
        if mode == 'neighbors_with_mask':
            return self._observe_neighbors(**kwarg)
        elif mode == 'neighbors_mask_secret_envinfo':
            assert (kwarg['agent_type'] == 'ci') or (self.mapping_period <= 0), 'Obs mode does not allow agent mapping'
            obs, mask = self._observe_neighbors(**kwarg)
            if show_agent_type_secrets(**self.secrete_args, **kwarg):
                secretes = self.secretes
            else:
                secretes = None
            return obs, mask, secretes, self.envinfo
        elif mode == 'neighbors':
            return self._observe_neighbors(**kwarg)[0]
        elif mode == 'matrix': 
            return self._observe_matrix(**kwarg)
        else:
            raise NotImplementedError(f'Mode {mode} is not implemented.')

    def sample(self, agent_type, mode, batch_size, horizon=None, last=False, **kwarg):
        if batch_size > len(self.history_queue):
            return

        if not last:
            eff_horizon = min(len(self.history_queue), horizon)
            pos_idx = np.random.choice(eff_horizon, batch_size)
        else:
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
                if self.envinfo_history is not None:
                    all_env_state = self.envinfo_history[:,episode_idx]
                    prev_env_state = all_env_state[:,:,:-1]
                    env_state = all_env_state[:,:,1:]
                else:
                    prev_env_state, env_state = None, None
                if show_agent_type_secrets(agent_type=agent_type, **self.secrete_args):
                    all_secrets = self.secretes_history[:, episode_idx]
                    prev_secrets = all_secrets[:,:,:-1]
                    secrets = all_secrets[:,:,1:]
                else:
                    secrets = None
                obs = (*obs, secrets, env_state)
                prev_obs = (*prev_obs, prev_secrets, prev_env_state)
        elif mode == 'matrix':
            prev_obs, obs = self._batch_matrix(
                episode_idx=episode_idx, agent_type=agent_type, **kwarg)
        else:
            raise NotImplementedError(f'Mode {mode} is not implemented.')


        actions = self.state_history[self.agent_types_idx[agent_type],:,episode_idx, 1:]
        rewards = self.reward_history[self.agent_types_idx[agent_type],:,episode_idx]
        if agent_type == 'ai':
            _ci_ai_map_history = self.ci_ai_map_history[:, episode_idx] \
                .unsqueeze(-1) \
                .repeat(1,1,self.episode_length)
            actions = th.gather(actions, 0, _ci_ai_map_history)
            rewards = th.gather(rewards, 0, _ci_ai_map_history)

        assert actions.max() <= self.n_actions - 1
        return prev_obs, obs, actions, rewards


    def _map_incoming(self, values):
        return {'ai': values['ai'][self.ai_ci_map], 'ci': values['ci']}

    def _map_outgoing(self, values):
        return {'ai': values['ai'][self.ci_ai_map], 'ci': values['ci']}

    def _get_rewards(self):
        ci_state = self.state[self.agent_types_idx['ci']]
        ai_state = self.state[self.agent_types_idx['ai']]

        ci_state_shifted = (ci_state + 1)
        ci_state_shifted_mapped = self.adjacency_matrix * (ci_state + 1) # [a,:] neighbors of agent a 

        conflicts = (ci_state_shifted[:, np.newaxis] == ci_state_shifted_mapped)

        ind_coordination = 1 - conflicts.any(dim=0).type(th.float)
        ind_catch = (ai_state == ci_state).type(th.float)

        metrics = {
            'ind_coordination': ind_coordination,
            'avg_coordination': ind_coordination.mean(0, keepdim=True).expand(self.n_agents),
            'ind_catch': ind_catch,
            'avg_catch': ind_catch.mean(0, keepdim=True).expand(self.n_agents),
            'rewarded': th.full(
                size=(self.n_agents,), 
                fill_value=(self.episode_step % self.reward_period) == 0, 
                dtype=th.float, device=self.device
            )
        }
        ci_reward = th.stack([metrics[k] * v for k,v in self.rewards_args['ci'].items()]).sum(0)*metrics['rewarded']
        ai_reward = th.stack([metrics[k] * v for k,v in self.rewards_args['ai'].items()]).sum(0)*metrics['rewarded']

        return ci_reward, ai_reward, metrics


    def step(self, actions):
        self.episode_step += 1

        assert actions['ci'].max() <= self.n_actions - 1
        assert actions['ai'].max() <= self.n_actions - 1
        assert actions['ai'].dtype == th.int64
        assert actions['ci'].dtype == th.int64


        if (self.episode_step + 1 == self.episode_length):
            done = True
        elif self.episode_step >= self.episode_length:
            raise ValueError('Environment is done already.')
        else:
            done = False


        self.state[self.agent_types_idx['ci']] = actions['ci']
        self.state[self.agent_types_idx['ai']] = actions['ai'][self.ai_ci_map]

        ci_reward, ai_reward, info = self._get_rewards()

        self.envinfo = self._get_envinfo(info)

        self._update_secrets()

        self.state_history[:, :, self.current_hist_idx, self.episode_step + 1] = self.state
        self.reward_history[self.agent_types_idx['ci'], :, self.current_hist_idx, self.episode_step] = ci_reward
        self.reward_history[self.agent_types_idx['ai'], :, self.current_hist_idx, self.episode_step] = ai_reward
        if self.envinfo_history is not None:
            self.envinfo_history[:,self.current_hist_idx, self.episode_step + 1] = self.envinfo
        if self.secretes_history is not None:
            self.secretes_history[:, self.current_hist_idx, self.episode_step + 1] = self.secretes

        rewards = {
            'ai': ai_reward[self.ci_ai_map],
            'ci': ci_reward
        }
        return rewards, done, info

    def close(self):
        pass
