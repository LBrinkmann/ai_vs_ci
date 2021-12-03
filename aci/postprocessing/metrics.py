from aci.postprocessing.utils import add_all, aggregation
from aci.utils.array_to_df import using_multiindex, map_columns


def agg_metrics(bin_size, metrics, metric_names, agents, episode, agent_types, **_):
    df = using_multiindex(metrics, ['episode', 'episode_step',
                                    'agent', 'agent_type', 'metric_name'], value_name='value')
    df = map_columns(df, agent=agents, metric_name=metric_names,
                     agent_type=agent_types, episode=episode.tolist())
    df = aggregation(df, quarter_column='episode_step', quarter_name='episode_part', bin_name='episode_bin',
                     bin_column='episode', bin_size=bin_size, agg_column='value', agg_func='mean')

    df = add_all(
        df, value_name='value', merge_column='agent', sum_name='all', agg_func='mean')

    column_order = [
        'agent', 'episode_bin', 'episode_part', 'agent_type', 'metric_name', 'value'
    ]
    cat_column = ['agent_type', 'agent', 'episode_bin', 'episode_part', 'metric_name']

    df = df.sort_values(cat_column).reset_index(drop=True)

    df[cat_column] = df[cat_column].astype('category')

    return df[column_order]
