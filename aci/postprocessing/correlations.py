import torch as th
from aci.postprocessing.utils import aggregation, add_all
from aci.utils.array_to_df import using_multiindex, map_columns


def create_correlations(actions, agents, agent_types, episode, max_delta, bin_size, **_):
    e_steps = actions.shape[1]
    diff = th.stack([
        actions[:, max_delta-i:e_steps-i] == actions[:, max_delta:]
        for i in range(max_delta)
    ], dim=-1).float()
    df = using_multiindex(diff, ['episode', 'episode_step',
                                 'agent', 'agent_type', 'delta'], value_name='value')
    df = map_columns(df, agent=agents, agent_type=agent_types, episode=episode.tolist())
    df = aggregation(df, quarter_column='episode_step', quarter_name='episode_part', bin_name='episode_bin',
                     bin_column='episode', bin_size=bin_size, agg_column='value', agg_func='mean')

    df = add_all(
        df, value_name='value', merge_column='agent', sum_name='all', agg_func='mean')

    column_order = [
        'agent', 'episode_bin', 'episode_part', 'agent_type', 'delta', 'value'
    ]
    cat_column = ['agent_type', 'agent', 'episode_bin', 'episode_part', 'delta']

    df = df.sort_values(cat_column).reset_index(drop=True)

    df[cat_column] = df[cat_column].astype('category')

    return df[column_order]
