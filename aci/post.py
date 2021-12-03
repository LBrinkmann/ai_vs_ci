"""Usage: train.py RUN_FOLDER [IN_FOLDER]

Arguments:
    RUN_FOLDER

Outputs:
    ....
"""

# from aci.envs.network_game import NetworkGame
from docopt import docopt
import os
from numpy.compat.py3k import is_pathlib_path
import pandas as pd
import numpy as np
import torch as th
import random
from multiprocessing import Manager, Pool
from aci.utils.io import load_yaml, ensure_dir
from aci.postprocessing.metrics import agg_metrics
from aci.postprocessing.pattern import create_pattern_df
from aci.postprocessing.tuples import match_tuples, agg_pattern_group, filter_top_pattern
from aci.postprocessing.entropy import calculate_pattern_metrics
from aci.postprocessing.correlations import create_correlations


N_JOBS = 16


def read_file(args):
    filename, fullfilename = args
    return (filename, pd.read_parquet(fullfilename))


def add_labels(df, labels):
    label_df = pd.DataFrame(data=labels, dtype='category', index=df.index)
    return pd.concat([label_df, df], axis=1)


def save_df(df, labels, name, metric_name, outdir):
    df = add_labels(df, labels)
    filename = os.path.join(outdir, f"{name}.{metric_name}.pgzip")
    print(f'Save {filename}.')
    df.to_parquet(filename)


def _preprocess_file(
        filename, mode, pattern_df, labels, outdir, name, correlations_args,
        tuple_args, metric_args):
    print(f'Start processing {filename}.')

    tensors = th.load(filename, map_location=th.device('cpu'))
    _labels = {**labels, 'mode': mode}

    ensure_dir(outdir)

    metrics_df = agg_metrics(**tensors, **metric_args)
    save_df(metrics_df, _labels, name, 'metrics', outdir)

    pattern_df = pd.concat(match_tuples(
        **tensors, pattern_df=pattern_df, **tuple_args))
    pattern_df = agg_pattern_group(pattern_df)

    top_pattern_df = filter_top_pattern(pattern_df)
    save_df(top_pattern_df, _labels, name, 'top_pattern', outdir)

    pattern_metrics_df = calculate_pattern_metrics(pattern_df)
    save_df(pattern_metrics_df, _labels, name, 'pattern_metrics', outdir)

    assert pattern_metrics_df.groupby(['agent_type', 'agent', 'episode_bin', 'episode_part',
                                       'pattern_length', 'metric_name', 'type'])['value'].count().max() == 1

    metrics_df = create_correlations(**tensors, **correlations_args)
    save_df(metrics_df, _labels, name, 'correlations', outdir)


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


def get_files(job_folder, out_folder):
    for mode in ['eval', 'train']:
        for f in os.listdir(os.path.join(job_folder, 'train', 'env', mode)):
            filename = os.path.join(job_folder, 'train', 'env', mode, f)
            outdir = os.path.join(out_folder, mode)
            name = os.path.splitext(f)[0]
            yield mode, name, filename, outdir


def _main(job_folder, out_folder, parse_args, labels, cores=1, seed=None):
    # Set seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        th.set_deterministic(True)

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

    print(in_folder)

    parameter_file = os.path.join(run_folder, 'post.yml')

    out_folder = os.path.join(run_folder, 'post')
    parameter = load_yaml(parameter_file)

    if in_folder:
        _main(job_folder=in_folder, out_folder=out_folder, cores=parameter['exec']['cores'],
              labels=parameter['labels'], **parameter['params'])
    else:
        # run_yml = os.path.join(run_folder, 'train.yml')
        # run_parameter = load_yaml(run_yml)
        # labels = run_parameter['labels']
        _main(job_folder=run_folder, out_folder=out_folder, cores=1,
              labels=parameter['labels'], **parameter['params'])


if __name__ == "__main__":
    main()
