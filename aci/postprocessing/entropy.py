from os import access
import pandas as pd
import numpy as np


def calculate_entropy(df):
    df = df.copy()
    groupby = ['agent_type', 'agent', 'episode_bin',
               'episode_part', 'pattern_length', 'type']
    prop_column = 'freq'

    df['value'] = -1 * df[prop_column] * np.log(df[prop_column])
    return df.groupby(groupby)['value'].sum().reset_index()


def calculate_kullback_leibler(df, across='agent'):
    """
        Calculates the kl divergence between each agent's pattern distribution and
        the one of all agents together.
    """
    df = df.copy()
    groupby = ['agent_type', 'episode_bin', 'agent',
               'episode_part', 'pattern_length', 'type']
    groupby = [c for c in groupby if c != across]
    w_all = df[across].isin(['all', -1])

    _df = df[~w_all].merge(df.loc[w_all, groupby + ['freq', 'name']],
                           on=groupby + ['name'], suffixes=['', '_all'])

    _df['value'] = _df['freq_all'] * \
        np.log((_df['freq_all'] / (_df['freq'] +
                                   np.finfo(float).eps)) + np.finfo(float).eps)

    _df = _df.groupby(groupby + [across],
                      observed=True)['value'].sum().reset_index()
    _df_all = _df.groupby(groupby, observed=True)['value'].mean().reset_index()
    _df_all[across] = df.loc[w_all, across].values[0]
    __df = pd.concat([_df, _df_all])
    assert not _df.duplicated(subset=groupby + [across], keep=False).any()
    assert not _df_all.duplicated(subset=groupby + [across], keep=False).any()
    assert not __df.duplicated(subset=groupby + [across], keep=False).any()
    return __df


def calculate_pattern_metrics(pattern_df):
    df_entropy = calculate_entropy(pattern_df)
    df_entropy['metric_name'] = 'entropy'
    df_kl = calculate_kullback_leibler(pattern_df, across='agent')
    df_kl['metric_name'] = 'js_div_agents'
    df_kl = calculate_kullback_leibler(pattern_df, across='episode_bin')
    df_kl['metric_name'] = 'js_div_episodes'

    df = pd.concat([df_entropy, df_kl])
    column_order = [
        'agent_type', 'agent', 'episode_bin', 'episode_part', 'pattern_length', 'metric_name', 'type', 'value'
    ]
    cat_column = ['agent_type', 'agent', 'episode_bin',
                  'episode_part', 'pattern_length', 'metric_name', 'type']

    assert not df.duplicated(subset=cat_column, keep=False).any()
    df[cat_column] = df[cat_column].astype('category')
    return df[column_order]
