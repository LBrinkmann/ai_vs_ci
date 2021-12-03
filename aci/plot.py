"""Usage: train.py RUN_FOLDER

Arguments:
    RUN_FOLDER

Outputs:
    ....
"""

from docopt import docopt
from numpy.lib.financial import ipmt
import pandas as pd
import traceback
import os
from multiprocessing import Manager, Pool
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import numpy as np
from aci.utils.df_utils import (
    selector, expand_selectors, summarize_df, preprocess)
from aci.utils.io import load_yaml, ensure_dir


def _plot(args):
    try:
        plot(args[1].dfs, **args[0])
    except Exception as ex:
        title = ' | '.join(f"{k}:{v}" for k, v in args[0]
                           ['selectors'].items() if not isinstance(v, list))
        print(f'Error plotting {title}.')
        print(''.join(traceback.format_exception(
            etype=type(ex), value=ex, tb=ex.__traceback__)))


def plot(
        dfs, filename, output_path, name, selectors, x, y, grid=[], dfs_baselines=None, grid_order={},
        hue=None, style=None, idx=None, x_label=None, y_label=None):

    df = dfs[filename]
    if dfs_baselines:
        df_baselines = dfs_baselines[filename]
        df = pd.concat([df, df_baselines])
    # import ipdb
    # ipdb.set_trace()
    # else:
    #     df_baselines = None

    w = selector(df, selectors)

    title = ' | '.join(
        f"{k}:{v}" for k, v in selectors.items() if not isinstance(v, list))

    print(f'start plotting {name}.{idx}')

    dfs = df[w].copy()

    if (hue is not None) and pd.api.types.is_numeric_dtype(dfs[hue].dtype):
        dfs[hue] = "#" + dfs[hue].astype(str)

    for col, iscat in (df.dtypes == 'category').iteritems():
        if iscat:
            dfs[col] = dfs[col].cat.remove_unused_categories()

    # assert indiviuallity
    groupby = [c for c in (grid + [hue, style, x]) if c is not None]

    dfs = dfs.dropna(subset=groupby)
    df_dup = dfs.duplicated(subset=groupby, keep=False)
    if df_dup.any():
        dup = dfs[df_dup].sort_values(groupby)
        print(dup.head())
        diff = dup.iloc[0] != dup.iloc[1]
        print(dup.loc[:, diff])
        raise ValueError('There are doublicates.')

    grid.sort(key=lambda l: dfs[l].nunique())

    grid = {n: g for g, n in zip(grid[::-1], ['col', 'row'])}

    # print(grid_order, grid)

    grid_order = {
        f'{k}_order': grid_order[v] if v in grid_order else sorted([n for n in dfs[v].unique() if not pd.isnull(n)])
        for k, v in grid.items()
    }

    # print(grid_order)

    other = {}
    if hue is not None:
        other['hue'] = hue
    if style is not None:
        other['style'] = style

    other_order = {
        f'{k}_order': sorted([n for n in dfs[v].unique() if not pd.isnull(n)])
        for k, v in other.items()
    }

    grid = sns.FacetGrid(
        data=dfs,
        **grid,
        **grid_order)

    grid.map_dataframe(
        single_plot,
        x=x,
        y=y,
        baseline=dfs_baselines is not None,
        # df_baselines=df_baselines,
        ** other,
        **other_order
    )
    grid.set_axis_labels(x_label, y_label)
    # import ipdb
    # ipdb.set_trace()

    # plt.tight_layout()
    top = 0.9 if len(grid.axes) >= 2 else 0.8
    plt.subplots_adjust(top=top, bottom=0.15)
    grid.fig.suptitle(title)
    grid.fig.patch.set_facecolor('white')
    grid.add_legend()
    if idx is not None:
        ensure_dir(os.path.join(output_path, name))
        filename = os.path.join(output_path, name, f"{idx}.png")
    else:
        filename = os.path.join(output_path, f"{name}.png")
    plt.savefig(filename, dpi=300)
    print(f'Saved {filename}')
    plt.close()


def single_plot(data, baseline, x, y, color, *args, **kwargs):
    if baseline:
        _data = data[data['baseline'].isnull()]
        baseline_data = data[~data['baseline'].isnull()]
        df_dup = baseline_data.duplicated(subset=[x, 'baseline'], keep=False)
        if df_dup.any():
            raise ValueError('There are doublicates in baseline.')
            # import ipdb
            # ipdb.set_trace()
            # baseline_hue = kwargs['hue'] if kwargs['hue'] in df_baseline.columns else None
            # baseline_style = kwargs['hue'] if kwargs['hue'] in df_baseline.columns else None

            # baseline_kwargs = {k: v for k, v in kwargs.items() if (v in df_baseline.columns)}
        sns.lineplot(
            data=baseline_data,
            x=x,
            y=y,
            # *args,
            # **kwargs,
            color='gray',
            lw=1,
            ci=None
        )
    else:
        _data = data
    sns.lineplot(
        data=_data,
        x=x,
        y=y,
        color=color,
        *args,
        **kwargs,
        lw=1,
        ci=None
    )


def apply_expand(dfs, selectors, filename, expand, **kwargs):
    for idx, sel in enumerate(expand_selectors(dfs[filename], selectors, expand=expand)):
        yield {'selectors': sel, 'filename': filename, **kwargs, 'idx': idx}


def create_plots_args(dfs, plots, **kwargs):
    for p in plots:
        for plot_args in apply_expand(dfs, **p, **kwargs):
            yield plot_args


def load_files(input_files, preprocess_args):
    input_files = {
        os.path.split(f)[-1].split('.')[-2]: f
        for f in input_files
    }

    dfs = {
        name: preprocess(pd.read_parquet(input_files[name]), **pa)
        for name, pa in preprocess_args.items()
    }

    # Summarize dataframe
    for name, df in dfs.items():
        summarize_df(df, name)
    return dfs


def merge_baselines(dfs):
    df_out = {}
    for bs_name, dfss in dfs.items():
        for df_name, df in dfss.items():
            if df_name not in df_out:
                df_out[df_name] = []
            df['baseline'] = bs_name
            df_out[df_name].append(df)
    df_out = {k: pd.concat(v) for k, v in df_out.items()}
    return df_out


def _main(
        *, input_files, baseline_files, clean, preprocess_args=None,
        preprocess_baseline_args=None, cores=1, plots, output_path):
    if clean:
        shutil.rmtree(output_path, ignore_errors=True)
    ensure_dir(output_path)

    print("Preprocess files.")
    dfs = load_files(input_files, preprocess_args)

    if baseline_files:
        print("Preprocess baseline.")
        dfs_baselines = {k: load_files(v, preprocess_baseline_args)
                         for k, v in baseline_files.items()}
        dfs_baselines = merge_baselines(dfs_baselines)
    else:
        dfs_baselines = None

    plot_args = list(
        create_plots_args(dfs, dfs_baselines=dfs_baselines, plots=plots, output_path=output_path))

    print(f"Make {len(plot_args)} plots.")

    mgr = Manager()
    ns = mgr.Namespace()
    ns.dfs = dfs

    data_args = list(zip(plot_args, [ns]*len(plot_args)))
    if cores > 1:
        pool = Pool(cores)
        pool.map(_plot, data_args)
    else:
        for da in data_args:
            _plot(da)


def main():
    arguments = docopt(__doc__)
    run_folder = arguments['RUN_FOLDER']
    parameter_file = os.path.join(run_folder, 'plot.yml')

    out_folder = os.path.join(run_folder, 'plot')
    ensure_dir(out_folder)
    parameter = load_yaml(parameter_file)

    input_files = os.listdir(os.path.join(run_folder, 'merge'))
    input_files = [os.path.join(run_folder, 'merge', f) for f in input_files]

    baselines = parameter.pop(
        'baselines') if 'baselines' in parameter else None

    if baselines:
        baseline_files = {k: os.listdir(os.path.join(
            v, 'merge')) for k, v in baselines.items()}
        baseline_files = {
            k: [os.path.join(baselines[k], 'merge', f) for f in v] for k, v in baseline_files.items()}
    else:
        baseline_files = None

    _main(output_path=out_folder, input_files=input_files,
          baseline_files=baseline_files, **parameter)


if __name__ == "__main__":
    main()
