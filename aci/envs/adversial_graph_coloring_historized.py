"""
Graph Coloring Environment
"""

import torch as th
import numpy as np
import collections
from .utils.graph import create_graph, determine_max_degree, pad_neighbors
    

class AdGraphColoringHist():
    def __init__(
            self, n_agents, n_actions, graph_args, episode_length, max_history, mapping_period, 
            network_period, rewards_args, device):
        self.episode_length = episode_length
        self.mapping_period = mapping_period
        self.network_period = network_period
        self.max_history = max_history
        self.graph_args = graph_args
        self.device = device
        self.rewards_args = rewards_args
        self.agent_types = ['ci', 'ai']
        self.agent_types_idx = {'ci': 0, 'ai': 1}
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.episode = -1
        self.max_degree = determine_max_degree(n_nodes=n_agents, **graph_args)

        self.init_history()

    def init_history(self):
        self.history_queue = collections.deque([], maxlen=self.max_history)
        self.state_history = th.empty(
            (len(self.agent_types), self.n_agents, self.max_history, self.episode_length + 1), dtype=th.int64, device=self.device)
        self.reward_history = th.empty(
            (len(self.agent_types), self.n_agents, self.max_history, self.episode_length), dtype=th.float32, device=self.device)
        self.adjacency_matrix_history = th.empty(
            (self.n_agents, self.max_history, self.n_agents), dtype=th.bool, device=self.device)

        self.neighbors_history = th.empty((self.n_agents, self.max_history, self.max_degree + 1), dtype=th.int64, device=self.device)
        self.neighbors_mask_history = th.empty((self.n_agents, self.max_history, self.max_degree + 1), dtype=th.bool, device=self.device)

        self.ci_ai_map_history = th.empty((self.n_agents, self.max_history), dtype=th.int64, device=self.device)

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
            self.adjacency_matrix = th.tensor(adjacency_matrix, dtype=th.bool, device=self.device)
            
            
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

        # add to history
        self.history_queue.appendleft(self.current_hist_idx)
        self.state_history[:, :, self.current_hist_idx, self.episode_step] = self.state
        self.adjacency_matrix_history[:, self.current_hist_idx] = self.adjacency_matrix
        self.neighbors_history[:, self.current_hist_idx] = self.neighbors
        self.neighbors_mask_history[:, self.current_hist_idx] = self.neighbors_mask
        self.ci_ai_map_history[:, self.current_hist_idx] = self.ci_ai_map

        self.info = {
            'graph': self.graph,
            'graph_info': self.graph_info,
            'ai_ci_map': self.ai_ci_map.tolist(),
            'ci_ai_map': self.ci_ai_map.tolist()
        }

    def to_dict(self):
        return self.info

    def _observe_historized_neighbors(self, agent_type, history_size):
        pass

    def _observe_neighbors(self, agent_type):
        ci_state = self.state[self.agent_types_idx['ci']]
        ci_state_ = ci_state.reshape(1,-1).repeat(self.n_agents,1)

        _neighbors = self.neighbors.clone()
        _neighbors[self.neighbors_mask] = 0
        obs = th.gather(ci_state_, 1, _neighbors)
        obs[self.neighbors_mask] = -1

        if agent_type == 'ci':
            return obs, self.neighbors_mask
        elif agent_type == 'ai':
            return obs[self.ci_ai_map], self.neighbors_mask[self.ci_ai_map]
        else:
            raise ValueError(f'Unkown agent_type {agent_type}')

    
    def _batch_neighbors(self, episode_idx, agent_type):
        ci_state = self.state_history[self.agent_types_idx['ci'], :, episode_idx]
        _ci_state = ci_state.permute(1,2,0).unsqueeze(0).repeat(self.n_agents,1,1,1)

        _mask = self.neighbors_mask_history[:, episode_idx]
        _neighbors = self.neighbors_history[:, episode_idx].clone()
        _neighbors[_mask] = 0
        _neighbors = _neighbors.unsqueeze(-2).repeat(1,1,self.episode_length + 1,1)
        _mask = _mask.unsqueeze(-2).repeat(1,1,self.episode_length + 1,1)

        obs = th.gather(_ci_state, -1, _neighbors)
        obs[_mask] = -1
        if agent_type == 'ci':
            return obs, _mask
        elif agent_type == 'ai':
            _ci_ai_map_history = self.ci_ai_map_history[:, episode_idx] \
                .unsqueeze(-1) \
                .unsqueeze(-1) \
                .repeat(1,1,obs.shape[2],obs.shape[3])
            return th.gather(obs, 0, _ci_ai_map_history), th.gather(_mask, 0, _ci_ai_map_history)

        else:
            raise ValueError(f'Unkown agent_type {agent_type}')


    def get_observation_shapes(self):
        return {
            'neighbors_with_mask': ((self.n_agents, self.max_degree + 1), (self.n_agents, self.max_degree + 1)),
            'neighbors': (self.n_agents, self.max_degree + 1)
        }


    def observe(self, mode, **kwarg):
        if mode == 'neighbors_with_mask':
            return self._observe_neighbors(**kwarg)
        elif mode == 'neighbors': 
            return self._observe_neighbors(**kwarg)[0]
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
        all_observations, all_mask = self._batch_neighbors(
            episode_idx=episode_idx, agent_type=agent_type, **kwarg)

        prev_observations = all_observations[:,:,:-1]
        observations = all_observations[:,:,1:]
        prev_mask = all_mask[:,:,:-1]
        mask = all_mask[:,:,1:]            

        actions = self.state_history[self.agent_types_idx[agent_type],:,episode_idx, 1:]
        rewards = self.reward_history[self.agent_types_idx[agent_type],:,episode_idx]
        if agent_type == 'ai':
            _ci_ai_map_history = self.ci_ai_map_history[:, episode_idx] \
                .unsqueeze(-1) \
                .repeat(1,1,self.episode_length)
            actions = th.gather(actions, 0, _ci_ai_map_history)
            rewards = th.gather(rewards, 0, _ci_ai_map_history)
        if mode == 'neighbors_with_mask':
            return (prev_observations, prev_mask), (observations, mask), actions, rewards
        elif mode == 'neighbors': 
            return prev_observations, observations, actions, rewards
        else:
            raise NotImplementedError(f'Mode {mode} is not implemented.')


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
        }

        ci_reward = th.stack([metrics[k] * v for k,v in self.rewards_args['ci'].items()]).sum(0)
        ai_reward = th.stack([metrics[k] * v for k,v in self.rewards_args['ai'].items()]).sum(0)

        return ci_reward, ai_reward, metrics


    def step(self, actions):
        self.episode_step += 1
        assert actions['ci'].max() <= self.n_actions - 1
        assert actions['ai'].max() <= self.n_actions - 1
        if (self.episode_step + 1 == self.episode_length):
            done = True
        elif self.episode_step >= self.episode_length:
            raise ValueError('Environment is done already.')
        else:
            done = False


        self.state[self.agent_types_idx['ci']] = actions['ci']
        self.state[self.agent_types_idx['ai']] = actions['ai'][self.ai_ci_map]

        ci_reward, ai_reward, info = self._get_rewards()

        self.state_history[:, :, self.current_hist_idx, self.episode_step + 1] = self.state
        self.reward_history[self.agent_types_idx['ci'], :, self.current_hist_idx, self.episode_step] = ci_reward
        self.reward_history[self.agent_types_idx['ai'], :, self.current_hist_idx, self.episode_step] = ai_reward


        
        rewards = {
            'ai': ai_reward[self.ci_ai_map],
            'ci': ci_reward
        }
        return rewards, done, info

    def close(self):
        pass
