import torch as th
from .utils.shift_observations import shift_obs


class TabularQ:
    def __init__(
            self, observation_shape, n_agents, env_info, gamma, alpha,
            q_start, device, share_table=False):
        n_actions = env_info['n_actions']
        max_neighbors = env_info['max_neighbors']
        self.device = device

        n_q_tables = 1 if share_table else n_agents
        self.q_table = th.ones((n_q_tables, n_actions, max_neighbors,
                                n_actions), device=device) * q_start
        self.n_agents = n_agents
        self.alpha = alpha
        self.gamma = gamma
        self.share_table = share_table

    def get_q(self, neighbor_id, self_id, **__):
        """ Retrieves q values for all possible actions.
        """

        if self.share_table:
            q_table_idxs = th.zeros(self_id.shape[:2], dtype=th.int64)
        else:
            q_table_idxs = th.arange(self.n_agents, dtype=th.int64).unsqueeze(
                0).unsqueeze(0).expand(*self_id.shape[:1], -1)
        q_value = self.q_table[q_table_idxs, self_id, neighbor_id]
        return q_value

    def init_episode(self, neighbor_id, self_id):

        pass

    def update(self, observations, actions, rewards):
        pass
        previous_obs, current_obs = shift_obs(observations)

        if self.share_table:
            q_table_idxs = th.zeros(previous_obs['self_id'].shape[:2], dtype=th.int64)
        else:
            q_table_idxs = th.arange(self.n_agents, dtype=th.int64).unsqueeze(
                0).unsqueeze(0).expand(*previous_obs['self_id'].shape[:1], -1)

        old_q_values = self.q_table[q_table_idxs,
                                    previous_obs['self_id'], previous_obs['neighbor_id'], actions]
        next_max_q_val = self.q_table[q_table_idxs,
                                      current_obs['self_id'], current_obs['neighbor_id'], ].max(-1)[0]
        new_q_value = (1 - self.alpha) * old_q_values + self.alpha * \
            (rewards + self.gamma * next_max_q_val)

        if self.share_table:
            # currently this is ill behaving. There could be two subagents with the same prev_obs_id and actions, and it
            # is not clear of which agent the value is taken.
            self.q_table[0, prev_observations_idx, actions] = new_q_value
        else:
            self.q_table[q_table_idxs, prev_observations_idx, actions] = new_q_value

        if writer and done:
            self.log(writer)

    def log(self, writer):
        pass
        # if writer.check_on(on='table'):
        #     print(f'log {str(writer.step)}')
        #     df = using_multiindex(self.q_table.numpy(), ['agents', 'obs_idx', 'action'])
        #     df = map_columns(df, obs_idx=self.lookup.keys())
        #     df = to_alphabete(df, ['agents'])

        #     df = pd.pivot_table(df, index=['agents', 'obs_idx'], columns='action')
        #     writer.add_table(name='qtable', df=df, sheet=str(writer.step))
