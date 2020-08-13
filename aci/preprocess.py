"""Usage: train.py PARAMETER_FILE

Arguments:
    RUN_PATH

Outputs:
    ....
"""

from docopt import docopt
import os
import pandas as pd
import multiprocessing
import itertools
from itertools import groupby
from aci.utils.io import load_yaml


N_JOBS = 16


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def read_file(args):
    filename, fullfilename = args
    return (filename, pd.read_parquet(fullfilename))


def find_files_with_extension(_dir, ext):
    for root, dirs, files in os.walk(_dir):
        for file in files:
            if file.endswith(ext):
                yield (file, os.path.join(root, file))



def binby(df, groupby=['mode', 'name'], bin_by='episode', bin_size={}, values='value'):
    df = df.reset_index()
    for k, v in bin_size.items():
        assert v % 2 == 0, 'episode_bin need to be an even number.'
        w = df['mode'] == k
        df.loc[w, bin_by] = (df.loc[w, bin_by] // v) * v + v // 2
    df = df.groupby([bin_by] + groupby)[values].mean()
    return df

def obj_to_category(df):
    df.loc[:,df.dtypes == 'object'] = df.loc[:,df.dtypes == 'object'].astype('category')
    return df


def mean_bin(df, bin_args):
    # mean over agents and episode_steps
    df['mean'] = df.mean(axis=1)
    # mean over episode_steps and episode bins
    df = binby(df, **bin_args, values=df.columns)
    df.columns.name = 'agents'
    sr = df.stack()
    sr.name = 'value'
    df = sr.reset_index()
    df = obj_to_category(df)
    return df


def max_freq(df, groupby=['mode', 'name'], bin_by='episode', bin_size=10, values='value'):
    assert bin_size % 2 == 0, 'episode_bin need to be an even number.'
    index_col = list(df.index.names)
    df = df.reset_index()
    df[bin_by] = (df[bin_by] // bin_size) * bin_size + bin_size // 2
    df = df.set_index(index_col)
    sr = df.stack()
    sr.name = 'color'
    sr = sr.groupby(['episode', 'mode', 'name', 'agents']).value_counts(normalize=True)
    sr.name = 'value'
    sr = sr.groupby(['episode', 'mode', 'name', 'agents']).max()
    df = sr.unstack()
    df['mean'] = df.mean(axis=1)
    sr = df.stack()
    sr.name = 'value'
    df = sr.reset_index()
    df = obj_to_category(df)
    return df


def parse_info(df, bin_args):
    return [mean_bin(df, bin_args)]


def parse_reward(df, bin_args):
    df = mean_bin(df, bin_args)
    df['agent_type'] = df['name']
    df['name'] = pd.Series('reward', index=df.index, dtype='category')
    return [df]


def parse_actions(df, bin_args):
    df = df.sort_index()
    df = df.astype('Int64')
    df_same = df.groupby(['episode', 'mode', 'name']).shift(0) == df.groupby(['episode', 'mode', 'name']).shift(1)
    df_same = df_same.astype('float')
    df_same = df_same.groupby(['episode', 'mode', 'name']).mean()
    df_same = mean_bin(df_same, bin_args)
    df_same['agent_type'] = df_same['name']
    df_same['name'] = pd.Series('stick_to_color', index=df_same.index, dtype='category')

    df_max_count = max_freq(df, bin_args)
    df_max_count['agent_type'] = df_max_count['name']
    df_max_count['name'] = pd.Series('max_freq', index=df_max_count.index, dtype='category')

    return [df_same, df_max_count]

parser = {
    'info': parse_info,
    'rewards': parse_reward,
    'actions': parse_actions
}


def parse(name, files, parse_args):
    df = pd.concat(pd.read_parquet(p) for n, p in files)
    return parser[name](df, **parse_args[name])

def main(exp_dir, output_file, parse_args, labels):
    files_g = find_files_with_extension(exp_dir, ".parquet")

    files_g = list(files_g)

    files_grouped = groupby(files_g, lambda f: f[0].split('.')[0])

    dfs = []
    for name, files in files_grouped:
        dfs.extend(parse(name, files, parse_args))
    df = pd.concat(dfs)
    df = obj_to_category(df)
    for k, v in labels.items():
        df[k] = pd.Series(v, index=df.index, **({'dtype': 'category'} if isinstance(v, str) else {}))
    df.to_parquet(output_file)


if __name__ == "__main__":
    arguments = docopt(__doc__)
    parameter_file = arguments['PARAMETER_FILE']
    exp_dir = os.path.dirname(parameter_file)
    output_folder = os.path.join(exp_dir, 'preprocess')
    ensure_dir(output_folder)
    output_file = os.path.join(output_folder, 'metrics.parquet')
    parameter = load_yaml(parameter_file)

    run_yml = os.path.join(exp_dir, 'train.yml')
    run_parameter = load_yaml(run_yml)
    labels = run_parameter['labels']

    exp_dir = os.path.join(exp_dir, parameter.pop('input_dir'))

    main(output_file=output_file, exp_dir=exp_dir, labels=labels, **parameter)
