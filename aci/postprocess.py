"""Usage: train.py RUN_FOLDER [IN_FOLDER]

Arguments:
    RUN_FOLDER

Outputs:
    ....
"""

import string
from docopt import docopt
import os
import pandas as pd
import numpy as np
import torch as th
from multiprocessing import Manager, Pool
# from itertools import groupby
from functools import partial
from itertools import product
from aci.utils.io import load_yaml, ensure_dir
from aci.utils.array_to_df import using_multiindex, map_columns, to_alphabete


N_JOBS = 16


def read_file(args):
    filename, fullfilename = args
    return (filename, pd.read_parquet(fullfilename))


def add_random(state, actions, agent_types, **other):
    random = th.randint_like(state, len(actions))
    _state = th.cat([state, random])
    _agent_types = agent_types + ['random_ci', 'random_ai']
    return {**other, 'agent_types': _agent_types, 'state': _state, 'actions': actions}


def add_labels(df, labels):
    label_df = pd.DataFrame(data=labels, dtype='category', index=df.index)
    return pd.concat([label_df, df], axis=1)


def save_df(df, labels, name, metric_name, outdir):
    df = add_labels(df, labels)
    filename = os.path.join(outdir, f"{name}.{metric_name}.parquet")
    print(f'Save {filename}.')
    df.to_parquet(filename)


def _preprocess_file(filename, mode, pattern_df, labels, outdir, name, correlations_args, tuple_args, metric_args):
    print(f'Start processing {filename}.')

    tensors = th.load(filename, map_location=th.device('cpu'))
    tensors = add_random(**tensors)
    _labels = {**labels, 'mode': mode}

    ensure_dir(outdir)

    pattern_df = pd.concat(match_tuples(**tensors, pattern_df=pattern_df, **tuple_args))
    pattern_df = agg_pattern_group(pattern_df)

    top_pattern_df = filter_top_pattern(pattern_df)
    save_df(top_pattern_df, _labels, name, 'top_pattern', outdir)

    pattern_metrics_df = calculate_pattern_metrics(pattern_df)
    save_df(pattern_metrics_df, _labels, name, 'pattern_metrics', outdir)

    assert pattern_metrics_df.groupby(['agent_types', 'agent', 'episode_bin', 'episode_part',
                                       'pattern_length', 'metric_name', 'type'])['value'].count().max() == 1

    correlations_df = correlations(**tensors, **correlations_args)
    save_df(correlations_df, _labels, name, 'correlations', outdir)

    metrics_df = agg_metrics(**tensors, **metric_args)
    save_df(metrics_df, _labels, name, 'metrics', outdir)


def agg_pattern_group(pattern_df):
    groupby = ['agent', 'episode_bin', 'episode_part', 'agent_types', 'pattern_length']
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
        'agent', 'episode_bin', 'episode_part', 'agent_types', 'pattern_length', 'name', 'type', 'count', 'freq'
    ]
    cat_column = ['agent', 'episode_bin', 'episode_part', 'agent_types', 'pattern_length', 'name']
    df[cat_column] = df[cat_column].astype('category')
    return df[column_order]


def filter_top_pattern(pattern_df):
    columns = ['agent', 'episode_bin', 'episode_part', 'agent_types', 'pattern_length', 'type']
    pattern_df = pattern_df.sort_values('count', ascending=False)
    pattern_df = pattern_df.groupby(columns).head(10)
    return pattern_df


# TODO: remove
_agents = [string.ascii_uppercase[i] for i in range(4)]
_actions = [string.ascii_uppercase[i] for i in range(4)]


# metrics


def local_aggregation(metric, neighbors, neighbors_mask):
    # state: agent_type, agents, batch, episode_step
    # neighbors: agents, batch, max_neighbors
    # neighbors_mask: agents, batch, max_neighbors
    n_source, n_agents, batch_size, episode_steps = metric.shape
    n_agents_2, batch_size_2, max_neighbors = neighbors.shape

    assert n_agents == n_agents_2
    assert n_agents == n_agents_2

    # permutation needed because we map neighbors on agents
    _metric = metric.permute(0, 2, 3, 1) \
        .unsqueeze(1).repeat(1, n_agents, 1, 1, 1)  # agents, batch, episode_step, neighbors

    _neighbors = neighbors.clone()
    _neighbors[neighbors_mask] = 0
    _neighbors = _neighbors.unsqueeze(0).unsqueeze(3).repeat(
        n_source, 1, 1, episode_steps, 1)  # agents, batch, episode_step, neighbors

    _neighbors_mask = neighbors_mask.unsqueeze(0).unsqueeze(3) \
        .repeat(n_source, 1, 1, episode_steps, 1)  # agents, batch, episode_step, neighbors

    # agent_type, agents, batch, episode_step, neighbors
    local_metrics = th.gather(_metric, -1, _neighbors)
    local_metrics[_neighbors_mask] = 0
    local_metrics = local_metrics.sum(-1) / (~_neighbors_mask).sum(-1)
    return local_metrics


def recalcuate_metrics(state, episode, neighbors, neighbors_mask, metrics, metric_names, agent_types, agents, **_):
    """
    We recaluclate the metrics for the random case.

    """

    at_map = {k: v for v, k in enumerate(agent_types)}

    observations = get_observations(state, neighbors, neighbors_mask)

    self_agent_types = [at_map['ci'], at_map['random_ci']]
    other_agent_types = [at_map['ci'], at_map['random_ci']]
    self_color = observations[self_agent_types][:, :, :, :, [0]]
    other_colors = observations[other_agent_types][:, :, :, :, 1:]

    coordination = (self_color != other_colors).all(-1).type(th.float)

    local_coordination = local_aggregation(coordination, neighbors, neighbors_mask)

    self_agent_types = [at_map['ci'], at_map['random_ci']]
    other_agent_types = [at_map['ai'], at_map['random_ai']]
    self_color = observations[self_agent_types, :, :, :, 0]
    other_color = observations[other_agent_types, :, :, :, 0]
    catch = (self_color == other_color).type(th.float)

    local_catch = local_aggregation(catch, neighbors, neighbors_mask)

    rec_metrics = th.stack([coordination, local_coordination, catch, local_catch], dim=-1)

    df = using_multiindex(
        rec_metrics, ['agent_source', 'agent', 'episode', 'episode_step', 'metric_name'], value_name='value')

    df['episode'] = df['episode'].map(pd.Series(episode))

    rec_metric_names = ['ind_coordination', 'local_coordination', 'ind_catch', 'local_catch']

    df = map_columns(df, agent=agents, metric_name=rec_metric_names,
                     agent_source=['trained', 'random'])

    # assert that newly calculated metrics are matching the old ones
    org_df = using_multiindex(
        metrics, ['agent', 'episode', 'episode_step', 'metric_name'], value_name='org_value')
    org_df['episode'] = org_df['episode'].map(pd.Series(episode))
    org_df = map_columns(org_df, agent=agents, metric_name=metric_names)
    test = df[df['agent_source'] == 'trained'].merge(org_df, how='left')

    assert np.isclose(test['value'], test['org_value']).all()

    return df


def agg_metrics(bin_size, **tensors):
    df = recalcuate_metrics(**tensors)
    # df = using_multiindex(metrics, ['agents', 'episode', 'episode_step', 'metric_name'], value_name='value')
    # df = map_columns(df, agents=agents, metric_name=metric_names)
    df = aggregation(df, quarter_column='episode_step', quarter_name='episode_part', bin_name='episode_bin',
                     bin_column='episode', bin_size=bin_size, agg_column='value', agg_func='mean')

    df = add_all(
        df, value_name='value', merge_column='agent', sum_name='all', agg_func='mean')

    column_order = [
        'agent', 'episode_bin', 'episode_part', 'agent_source', 'metric_name', 'value'
    ]
    cat_column = ['agent_source', 'agent', 'episode_bin', 'episode_part', 'metric_name']

    df[cat_column] = df[cat_column].astype('category')

    return df[column_order]


# match_tuples


def create_tuples(m, length):
    episode_steps = m.shape[-1]
    tuples = th.stack([m[:, :, :, i:(episode_steps + 1 - length + i)]
                       for i in range(length)], dim=-1)
    return tuples


def tuple_to_multiindex(A, columns):
    shape = A.shape[:-1]
    length = A.shape[-1]
    index = pd.MultiIndex.from_product([range(s) for s in shape], names=columns)
    action_ids = [tuple(a) for a in A.reshape(-1, length).tolist()]
    df = pd.DataFrame({'action_ids': action_ids}, index=index).reset_index()
    return df


def aggregation(df, quarter_column, quarter_name, bin_name, bin_column, bin_size, agg_column=None, agg_func='count'):
    columns = list(set(df.columns) - set((quarter_column, bin_column, agg_column))) + [bin_name]

    # create column to aggregate first and last quarter seperatly
    max_episode_step = df[quarter_column].max()
    df[quarter_name] = 'middle'
    df.loc[df[quarter_column] < max_episode_step * 0.25, quarter_name] = 'start'
    df.loc[df[quarter_column] > max_episode_step * 0.75, quarter_name] = 'end'

    # create column to bin episodes
    df[bin_name] = df[bin_column] // bin_size * bin_size + bin_size // 2
    if agg_column is None:
        df['count'] = 1
        agg_column = 'count'

    # aggreation with quarters seperatly
    df_agg = df.groupby(columns + [quarter_name])[agg_column].agg(agg_func)
    df_agg = df_agg.reset_index()

    # aggreation with full data
    df_all_agg = df.groupby(columns)[agg_column].agg(agg_func)
    df_all_agg = df_all_agg.reset_index()
    df_all_agg[quarter_name] = 'full'

    df = pd.concat([df_agg, df_all_agg])
    return df


def add_all(df, value_name, merge_column, sum_name, agg_func):
    groupby_columns = list(set(df.columns) - set([value_name, merge_column]))
    sum_df = df.groupby(groupby_columns)[value_name].agg(agg_func).reset_index()
    sum_df[merge_column] = sum_name
    return pd.concat([df, sum_df])


def match_tuples(state, pattern_df, agent_types, agents, actions, episode, bin_size, **_):
    for length, this_pattern_df in pattern_df.groupby("pattern_length"):
        print(f'Create tuple of {length}')

        # stack state tensor, such that we optain tuples
        tuples = create_tuples(state, length)

        # turn tensor in to dataframe, make dims to columns
        tuple_df = tuple_to_multiindex(tuples, ['agent_types', 'agent', 'episode', 'episode_step'])
        tuple_df['episode'] = tuple_df['episode'].map(pd.Series(episode))

        # to replace the tuple of action_ids with the pattern_id might speed up things
        tuple_df = this_pattern_df[['pattern_id', 'action_ids']].merge(tuple_df)
        tuple_df['pattern_id'] = tuple_df['pattern_id'].astype('category')
        tuple_df = tuple_df.drop(columns='action_ids')

        # make binsize a parameter, should depend on length
        tuple_df = aggregation(
            tuple_df, quarter_column='episode_step', quarter_name='episode_part', bin_name='episode_bin',
            bin_column='episode', bin_size=bin_size)

        # have agent and agent_types as strings
        tuple_df = map_columns(
            tuple_df, agent_types=agent_types, agent=agents)

        # adding total count of all agents
        tuple_df = add_all(
            tuple_df, value_name='count', merge_column='agent', sum_name='all', agg_func='sum')

        norm = tuple_df.groupby(['agent_types', 'agent', 'episode_bin', 'episode_part'])[
            'count'].transform('sum')
        tuple_df['freq'] = tuple_df['count'] / norm

        # in_group_norm = tuple_df.groupby(
        #     ['agent_types', 'agent', 'episode_bin', 'episode_part', 'pattern_group_name'])['count'].transform('sum')
        # tuple_df['in_group_freq'] = tuple_df['count'] / in_group_norm

        tuple_df = this_pattern_df.merge(tuple_df)
        tuple_df['pattern_length'] = length
        column_order = [
            'agent', 'episode_bin', 'episode_part', 'agent_types', 'pattern_length',
            'pattern_id',  'pattern_name', 'pattern_group_name', 'count', 'freq'
        ]
        cat_column = ['agent_types', 'agent', 'episode_bin', 'episode_part',
                      'pattern_length', 'pattern_id', 'pattern_name', 'pattern_group_name']

        tuple_df[cat_column] = tuple_df[cat_column].astype('category')

        yield tuple_df[column_order]


def calculate_entropy(df):
    df = df.copy()
    groupby = ['agent_types', 'agent', 'episode_bin', 'episode_part', 'pattern_length', 'type']
    prop_column = 'freq'

    df['value'] = -1 * df[prop_column] * np.log(df[prop_column])
    return df.groupby(groupby)['value'].sum().reset_index()


def calculate_kullback_leibler(df, ):
    """
        Calculates the kl divergence between each agent's pattern distribution and
        the one of all agents together.
    """
    df = df.copy()
    groupby = ['agent_types', 'episode_bin', 'episode_part', 'pattern_length', 'type']
    w_all = df['agent'] == 'all'

    _df = df[~w_all].merge(df.loc[w_all, groupby + ['freq', 'name']],
                           on=groupby + ['name'], suffixes=['', '_all'])

    _df['value'] = _df['freq_all'] * \
        np.log((_df['freq_all'] / (_df['freq'] + np.finfo(float).eps)) + np.finfo(float).eps)

    _df = _df.groupby(groupby + ['agent'], observed=True)['value'].sum().reset_index()
    _df_all = _df.groupby(groupby, observed=True)['value'].mean().reset_index()
    _df_all['agent'] = 'all'
    __df = pd.concat([_df, _df_all])
    assert not _df.duplicated(subset=groupby + ['agent'], keep=False).any()
    assert not _df_all.duplicated(subset=groupby + ['agent'], keep=False).any()
    assert not __df.duplicated(subset=groupby + ['agent'], keep=False).any()
    return __df


def calculate_pattern_metrics(pattern_df):
    df_entropy = calculate_entropy(pattern_df)
    df_entropy['metric_name'] = 'entropy'
    df_kl = calculate_kullback_leibler(pattern_df)
    df_kl['metric_name'] = 'kl_div'

    df = pd.concat([df_entropy, df_kl])
    column_order = [
        'agent_types', 'agent', 'episode_bin', 'episode_part', 'pattern_length', 'metric_name', 'type', 'value'
    ]
    cat_column = ['agent_types', 'agent', 'episode_bin',
                  'episode_part', 'pattern_length', 'metric_name', 'type']

    assert not df.duplicated(subset=cat_column, keep=False).any()

    df[cat_column] = df[cat_column].astype('category')
    return df[column_order]


# create pattern df

def de_duplicate(seq):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def normalize_pattern(pattern, actions):
    mapping = {name: actions[idx] for idx, name in enumerate(de_duplicate(pattern))}
    return ''.join(mapping[p] for p in pattern)


def create_single_pattern_df(length, actions):
    pattern_name = list(''.join(n) for n in product(actions, repeat=length))
    df = pd.DataFrame(
        {
            'action_ids': list(product(range(len(actions)), repeat=length)),
            'pattern_name': pattern_name,
            'pattern_length': length,
        },
    )
    return df


def create_pattern_df(min_length, max_length, actions):
    """
    Create tuples from the outer product of avaible actions.
    Tuples which are equivalent upon permutation of action names are
    assign the same pattern group.

    Parameters
    ----------
    min_length : int
        minimum length of tuple
    max_length : int
        maximum length of tuple
    actions : list
        name of actions

    Returns
    -------
    pd.DataFrame
        Dataframe with patterns. Columns are as follows:
        ==================  ==========================================
        pattern_id          Id of pattern.
        pattern_length      Length of pattern.
        pattern_name        Name of pattern as joined as action names.
        pattern_group_name  Pattern group.
        action_ids          Tuple with action ids of the pattern.
        ==================  ==========================================

    """
    pattern_dfs = [
        create_single_pattern_df(length, actions)
        for length in range(min_length, max_length + 1)
    ]
    pattern_df = pd.concat(pattern_dfs)
    pattern_df['pattern_id'] = np.arange(len(pattern_df))
    pattern_df['pattern_group_name'] = pattern_df['pattern_name'].map(
        lambda p: normalize_pattern(p, actions))
    category_columns = ['pattern_name', 'pattern_group_name']
    pattern_df[category_columns] = pattern_df[category_columns].astype('category')
    column_order = [
        'pattern_id', 'pattern_length', 'pattern_name',
        'pattern_group_name', 'action_ids'
    ]
    return pattern_df[column_order]


def match(m, ref):
    ref_ = ref.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
    matches = (m[np.newaxis] == ref_).all(-1).type(th.int).argmax(0)
    return matches


def get_observations(state, neighbors, neighbors_mask, **_):
    # state: agent_type, agents, batch, episode_step
    # neighbors: agents, batch, max_neighbors
    # neighbors_mask: agents, batch, max_neighbors
    n_agent_types, n_agents, batch_size, episode_steps = state.shape
    n_agents_2, batch_size_2, max_neighbors = neighbors.shape

    assert n_agents == n_agents_2
    assert n_agents == n_agents_2

    # permutation needed because we map neighbors on agents
    _state = state.permute(0, 2, 3, 1) \
        .unsqueeze(1).repeat(1, n_agents, 1, 1, 1)  # agent_type, agents, batch, episode_step, neighbors

    _neighbors = neighbors.clone()
    _neighbors[neighbors_mask] = 0
    _neighbors = _neighbors.unsqueeze(0).unsqueeze(3).repeat(
        n_agent_types, 1, 1, episode_steps, 1)  # agent_type, agents, batch, episode_step, neighbors

    _neighbors_mask = neighbors_mask \
        .unsqueeze(0).unsqueeze(3) \
        .repeat(n_agent_types, 1, 1, episode_steps, 1)  # agent_type, agents, batch, episode_step, neighbors

    # agent_type, agents, batch, episode_step, neighbors
    observations = th.gather(_state, -1, _neighbors)
    observations[_neighbors_mask] = -1
    return observations


# TODO:
# add
# degree_centrality(neighbors_mask)
#
def correlations(state, neighbors, neighbors_mask, episode, max_delta, bin_size, agent_types, agents, correlations, **_):
    observations = get_observations(state, neighbors, neighbors_mask)
    at_map = {k: v for v, k in enumerate(agent_types)}
    all_df = []
    for c in correlations:
        df_neighbor = neighbor_correlations(
            observations, neighbors_mask, episode=episode, at_map=at_map, max_delta=max_delta,
            bin_size=bin_size, **c)
        all_df.append(df_neighbor)
        df_self = node_correlations(
            observations, episode=episode, at_map=at_map, max_delta=max_delta,
            bin_size=bin_size, **c)
        all_df.append(df_self)
    df = pd.concat(all_df)

    # have agent and agent_types as strings
    df = map_columns(df, agent=agents)

    # adding total count of all agents
    df = add_all(
        df, value_name='value', merge_column='agent', sum_name='all', agg_func='mean')

    column_order = [
        'agent', 'episode_bin', 'episode_part', 'other', 'node', 'delta_t', 'metric_name', 'value']
    cat_column = ['other', 'node', 'agent', 'delta_t',
                  'episode_bin', 'episode_part', 'metric_name']

    df[cat_column] = df[cat_column].astype('category')
    return df[column_order]


def degree_centrality(neighbors_mask):
    degree_centrality = (~neighbors_mask[:, :, 1:]
                         ).sum(-1).type(th.float) / (neighbors_mask.shape[0] - 1)
    return degree_centrality  # agents, batch


def node_correlation(observations, delta_t, self_agent_type, other_agent_type):
    """
    Simple metric
        #same_neigbhors / #sum_neighbors

    input:
        observations: [agent_type, agents, batch, episode_step, neighbors]
        mask: [agent_type, agents, batch, episode_step, neighbors]
        delta_t: compare self color at time t with other colors at time t - delta_t
        self_agent_type: agent type of self
        other_agent_type: agent type of neighbors
    """
    if delta_t != 0:
        self_color = observations[self_agent_type, :, :, delta_t:, 0]
        other_color = observations[other_agent_type, :, :, :-delta_t, 0]
    else:
        self_color = observations[self_agent_type, :, :, :, 0]
        other_color = observations[other_agent_type, :, :, :, 0]
    same_color = (self_color == other_color).type(th.float)
    return same_color  # agents, batch, episodes


def neighbor_correlation(observations, mask, delta_t, self_agent_type, other_agent_type):
    """
    Simple metric
        #same_neigbhors / #sum_neighbors

    input:
        observations: [agent_type, agents, batch, episode_step, neighbors]
        mask: [agent_type, agents, batch, episode_step, neighbors]
        delta_t: compare self color at time t with other colors at time t - delta_t
        self_agent_type: agent type of self
        other_agent_type: agent type of neighbors
    """
    if delta_t != 0:
        # remove the first delta_t datapoints in each episode
        self_color = observations[self_agent_type, :, :, delta_t:, [0]]
        # remove the last delta_t datapoints in each episode
        other_colors = observations[other_agent_type, :, :, :-delta_t, 1:]
    else:
        # remove the first delta_t datapoints in each episode
        self_color = observations[self_agent_type, :, :, :, [0]]
        # remove the last delta_t datapoints in each episode
        other_colors = observations[other_agent_type, :, :, :, 1:]

    same_color_count = (self_color == other_colors).type(th.float).sum(-1)

    neighbor_count = (~mask[:, :, 1:]).sum(-1).type(th.float).unsqueeze(-1)

    same_color_fraction = same_color_count / neighbor_count  # agents, batch, episodes

    return same_color_fraction


def x_correlations(*args, episode, other, node, at_map, max_delta, bin_size, function, metric_name):
    all_df = []
    for delta_t in range(max_delta):
        arr = function(*args, other_agent_type=at_map[other],
                       self_agent_type=at_map[node], delta_t=delta_t)
        df = using_multiindex(arr, ['agent', 'episode', 'episode_step'], value_name='value')
        df['episode'] = df['episode'].map(pd.Series(episode))
        df['delta_t'] = delta_t
        all_df.append(df)
    df = pd.concat(all_df)
    df = aggregation(df, quarter_column='episode_step', quarter_name='episode_part', bin_name='episode_bin',
                     bin_column='episode', bin_size=bin_size, agg_column='value', agg_func='mean')
    df['other'] = other
    df['node'] = node
    df['metric_name'] = metric_name
    return df


neighbor_correlations = partial(
    x_correlations, function=neighbor_correlation, metric_name='neighbor_correlation')
node_correlations = partial(x_correlations, function=node_correlation,
                            metric_name='node_correlation')


def preprocess_file(args):
    return _preprocess_file(**args)


def get_files(_dir):
    return [
        o
        for o in os.listdir(_dir)
        if os.path.isdir(os.path.join(_dir, o))
    ]


def get_subdirs(_dir):
    return [
        o
        for o in os.listdir(_dir)
        if os.path.isdir(os.path.join(_dir, o))
    ]


def get_labels(job_folder):
    run_yml = os.path.join(job_folder, 'train.yml')
    run_parameter = load_yaml(run_yml)
    labels = run_parameter['labels']
    return labels


def get_files(job_folder, out_folder):
    for mode in ['eval', 'train']:
        for f in os.listdir(os.path.join(job_folder, 'train', 'env', mode)):
            filename = os.path.join(job_folder, 'train', 'env', mode, f)
            outdir = os.path.join(out_folder, mode)
            name = os.path.splitext(f)[0]
            yield mode, name, filename, outdir


def _main(job_folder, out_folder, parse_args, labels, cores=1):

    pattern_df = create_pattern_df(**parse_args['pattern_args'])

    arg_list = [
        {
            'mode': mode, 'filename': filename, 'pattern_df': pattern_df, 'labels': labels,
            'outdir': outdir, 'name': name, **parse_args[mode]
        }
        for mode, name, filename, outdir in get_files(job_folder, out_folder)
    ]

    if cores == 1:
        for al in arg_list:
            preprocess_file(al)
    else:
        pool = Pool(cores)
        pool.map(preprocess_file, arg_list)


def main():
    arguments = docopt(__doc__)
    run_folder = arguments['RUN_FOLDER']
    in_folder = arguments['IN_FOLDER']

    parameter_file = os.path.join(run_folder, 'preprocess.yml')

    out_folder = os.path.join(run_folder, 'preprocess')
    parameter = load_yaml(parameter_file)

    if in_folder:
        _main(job_folder=in_folder, out_folder=out_folder, cores=parameter['exec']['cores'],
              labels=parameter['labels'], **parameter['params'])
    else:
        run_yml = os.path.join(run_folder, 'train.yml')
        run_parameter = load_yaml(run_yml)
        labels = run_parameter['labels']
        _main(job_folder=run_folder, out_folder=out_folder, cores=1,
              labels=labels, **parameter)


if __name__ == "__main__":
    main()
