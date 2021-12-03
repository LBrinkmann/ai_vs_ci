import pandas as pd
import numpy as np
import torch as th
from aci.utils.array_to_df import map_columns
from aci.postprocessing.utils import add_all, aggregation

# match_tuples


def create_tuples(m, length):
    episode_steps = m.shape[1]
    tuples = th.stack([m[:, i:(episode_steps + 1 - length + i)]
                       for i in range(length)], dim=-1)
    return tuples


def tuple_to_multiindex(A, columns):
    shape = A.shape[:-1]
    length = A.shape[-1]
    index = pd.MultiIndex.from_product(
        [range(s) for s in shape], names=columns)
    action_ids = [tuple(a) for a in A.reshape(-1, length).tolist()]
    df = pd.DataFrame({'action_ids': action_ids}, index=index).reset_index()
    return df


def match_tuples(actions, pattern_df, agent_types, agents, episode, bin_size, **_):
    for length, this_pattern_df in pattern_df.groupby("pattern_length"):
        print(f'Create tuple of {length}')

        # stack state tensor, such that we optain tuples
        tuples = create_tuples(actions, length)

        # turn tensor in to dataframe, make dims to columns
        tuple_df = tuple_to_multiindex(
            tuples, ['episode', 'episode_step', 'agent', 'agent_type'])

        # have agent and agent_types as strings
        tuple_df = map_columns(tuple_df, agent_type=agent_types,
                               agent=agents, episode=episode.tolist())

        # to replace the tuple of action_ids with the pattern_id might speed up things
        tuple_df = this_pattern_df[[
            'pattern_id', 'action_ids']].merge(tuple_df)
        tuple_df['pattern_id'] = tuple_df['pattern_id'].astype('category')
        tuple_df = tuple_df.drop(columns='action_ids')

        # make binsize a parameter, should depend on length
        tuple_df = aggregation(
            tuple_df, quarter_column='episode_step', quarter_name='episode_part', bin_name='episode_bin',
            bin_column='episode', bin_size=bin_size)

        # adding total count of all agents
        tuple_df = add_all(
            tuple_df, value_name='count', merge_column='agent', sum_name='all', agg_func='sum')

        # adding total count of episodes
        tuple_df = add_all(
            tuple_df, value_name='count', merge_column='episode_bin', sum_name=-1, agg_func='sum')

        norm = tuple_df.groupby(['agent_type', 'agent', 'episode_bin', 'episode_part'])[
            'count'].transform('sum')
        tuple_df['freq'] = tuple_df['count'] / norm

        # in_group_norm = tuple_df.groupby(
        #     ['agent_types', 'agent', 'episode_bin', 'episode_part', 'pattern_group_name'])['count'].transform('sum')
        # tuple_df['in_group_freq'] = tuple_df['count'] / in_group_norm

        tuple_df = this_pattern_df.merge(tuple_df)
        tuple_df['pattern_length'] = length
        column_order = [
            'agent', 'episode_bin', 'episode_part', 'agent_type', 'pattern_length',
            'pattern_id',  'pattern_name', 'pattern_group_name', 'count', 'freq'
        ]
        cat_column = ['agent_type', 'agent', 'episode_bin', 'episode_part',
                      'pattern_length', 'pattern_id', 'pattern_name', 'pattern_group_name']

        tuple_df[cat_column] = tuple_df[cat_column].astype('category')

        yield tuple_df[column_order]


def agg_pattern_group(pattern_df):
    groupby = ['agent', 'episode_bin', 'episode_part',
               'agent_type', 'pattern_length']
    pattern_group_df = pattern_df.groupby(
        groupby + ['pattern_group_name'])['count'].sum().reset_index()
    norm = pattern_group_df.groupby(groupby)['count'].transform('sum')
    pattern_group_df['freq'] = pattern_group_df['count'] / norm
    pattern_group_df['name'] = pattern_group_df['pattern_group_name']
    pattern_group_df['type'] = 'pattern_group'
    pattern_df['name'] = pattern_df['pattern_name']
    pattern_df['type'] = 'pattern'
    df = pd.concat([pattern_df, pattern_group_df])
    column_order = [
        'agent', 'episode_bin', 'episode_part', 'agent_type', 'pattern_length', 'name', 'type', 'count', 'freq'
    ]
    cat_column = ['agent', 'episode_bin', 'episode_part',
                  'agent_type', 'pattern_length', 'name', 'type']
    df[cat_column] = df[cat_column].astype('category')
    return df[column_order]


def filter_top_pattern(pattern_df):
    columns = ['agent', 'episode_bin', 'episode_part',
               'agent_type', 'pattern_length', 'type']
    median_count = pattern_df.groupby(columns)['count'].transform('median')
    pattern_df = pattern_df[pattern_df['count'] > median_count]
    pattern_df = pattern_df.sort_values(columns).reset_index(drop=True)
    return pattern_df
