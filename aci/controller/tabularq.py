import torch as th
from .utils.shift_observations import shift_obs


class TabularQ:
    def __init__(
            self, observation_shape, env_info, gamma, alpha,
            q_start, device, agent_type, avg_table=False):
        n_actions = env_info['n_actions']
        max_neighbors = env_info['max_neighbors']
        n_nodes = env_info['n_nodes']
        self.device = device
        self.q_table = th.ones((n_nodes, n_actions, max_neighbors,
                                n_actions), device=device) * q_start
        self.q_table_idx = th.arange(n_nodes, dtype=th.int64, device=self.device)
        self.alpha = alpha
        self.gamma = gamma
        self.avg_table = avg_table

    def get_q(self, neighbor_id, self_id, **__):
        """
        Retrieves q values for all possible actions.
        """
        import ipdb
        ipdb.set_trace()
        neighbor_id = neighbor_id.squeeze(0).squeeze(0)
        self_id = self_id.squeeze(0).squeeze(0)
        q_value = self.q_table[self.q_table_idx, self_id, neighbor_id]
        return q_value.unsqueeze(0).unsqueeze(0)

    def init_episode(self, observation, action, reward, episode):
        self.prev_self_id = observation['self_id'].squeeze(0).squeeze(0)
        self.prev_neighbor_id = observation['neighbor_id'].squeeze(0).squeeze(0)

    def update(self, observation, action, reward, done, step, episode):
        current_self_id = observation['self_id'].squeeze(0).squeeze(0)
        current_neighbor_id = observation['neighbor_id'].squeeze(0).squeeze(0)
        action = action.squeeze(0).squeeze(0)
        reward = reward.squeeze(0).squeeze(0)

        old_q_values = self.q_table[self.q_table_idx,
                                    self.prev_self_id, self.prev_neighbor_id, action]
        next_max_q_val = self.q_table[self.q_table_idx,
                                      current_self_id, current_neighbor_id, ].max(-1)[0]
        temporal_difference = self.alpha * (reward + self.gamma * next_max_q_val - old_q_values)

        _temporal_difference = th.zeros_like(self.q_table)

        _temporal_difference[self.q_table_idx, self.prev_self_id,
                             self.prev_neighbor_id, action] = temporal_difference

        if self.avg_table:
            self.q_table += _temporal_difference.sum(0, keep_dim=True)
        else:
            self.q_table += _temporal_difference

        self.prev_self_id = current_self_id
        self.prev_neighbor_id = current_neighbor_id

    def log(self, writer):
        pass
        # if writer.check_on(on='table'):
        #     print(f'log {str(writer.step)}')
        #     df = using_multiindex(self.q_table.numpy(), ['agents', 'obs_idx', 'action'])
        #     df = map_columns(df, obs_idx=self.lookup.keys())
        #     df = to_alphabete(df, ['agents'])

        #     df = pd.pivot_table(df, index=['agents', 'obs_idx'], columns='action')
        #     writer.add_table(name='qtable', df=df, sheet=str(writer.step))
