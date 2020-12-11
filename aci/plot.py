"""Usage: train.py RUN_FOLDER

Arguments:
    RUN_FOLDER

Outputs:
    ....
"""

from docopt import docopt
import pandas as pd
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
    plot(args[1].dfs, **args[0])


def plot(
    dfs, filename, output_path, name, selectors, x, y, grid=[], baseline=None,
    hue=None, style=None, idx=None):

    df = dfs[filename]
    w = selector(df, selectors)

    title = ' | '.join(f"{k}:{v}" for k, v in selectors.items() if not isinstance(v, list))

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
    w_baseline = dfs[baseline['column']] == baseline['value']
    df_wo_baseline = dfs[~w_baseline]
    df_dup = df_wo_baseline.duplicated(subset=groupby, keep=False)
    if df_dup.any():
        dup = df_wo_baseline[df_dup].sort_values(groupby)
        print(dup.head())
        diff = dup.iloc[0] != dup.iloc[1]
        print(df_nb.loc[:,diff])
        raise ValueError('There are doublicates.')

    grid.sort(key=lambda l: dfs[l].nunique())

    grid = {n: g for g, n in zip(grid[::-1], ['col','row'])}

    grid_order = {
        f'{k}_order': sorted([n for n in dfs[v].unique() if not pd.isnull(n)])
        for k, v in grid.items()
    }

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
        baseline=baseline,
        **other,
        **other_order
    )

    plt.subplots_adjust(top=0.9)
    grid.fig.suptitle(title)
    grid.fig.patch.set_facecolor('white')
    grid.add_legend()
    if idx is not None:
        ensure_dir(os.path.join(output_path, name))
        filename = os.path.join(output_path, name, f"{idx}.png")
    else:
        filename = os.path.join(output_path, f"{name}.png")
    plt.savefig(filename)
    print(f'Saved {filename}')
    plt.close()


def single_plot(data, baseline, x, y, color, *args, **kwargs):
    w_baseline = data[baseline['column']] == baseline['value']
    baseline_data = data[w_baseline]
    other_data = data[~w_baseline]

    sns.lineplot(
        data=baseline_data,
        x=x,
        y=y,
        *args,
        **kwargs,
        color='gray',
        lw=0.5,
        ci=None
    )
    sns.lineplot(
        data=other_data,
        x=x,
        y=y,
        color=color,
        *args,
        **kwargs,
        lw=2,
        ci=None
    )


def apply_expand(dfs, selectors, filename, expand, **kwargs):
    for idx, sel in enumerate(expand_selectors(dfs[filename], selectors, expand=expand)):
        yield {'selectors': sel, 'filename': filename, **kwargs, 'idx': idx}


def create_plots_args(dfs, plots, **kwargs):
    for p in plots:
        for plot_args in apply_expand(dfs, **p, **kwargs):
            yield plot_args


def _main(*, input_files, clean, preprocess_args=None, plots, output_path):
    if clean:
        shutil.rmtree(output_path, ignore_errors=True)
    ensure_dir(output_path)

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

    plot_args = list(create_plots_args(dfs, plots=plots, output_path=output_path))

    print(f"Make {len(plot_args)} plots.")

    mgr = Manager()
    ns = mgr.Namespace()
    ns.dfs = dfs
    pool = Pool(20)

    # print(plot_args[0])
    # print(plot_args[1])

    data_args = list(zip(plot_args, [ns]*len(plot_args)))
    pool.map(_plot, data_args)
    # _plot(data_args[0])
    # _plot(data_args[1])


def main():
    arguments = docopt(__doc__)
    run_folder = arguments['RUN_FOLDER']
    parameter_file = os.path.join(run_folder, 'plot.yml')

    out_folder = os.path.join(run_folder, 'plot')
    ensure_dir(out_folder)
    parameter = load_yaml(parameter_file)

    input_files = os.listdir(os.path.join(run_folder, 'merge'))
    input_files = [os.path.join(run_folder, 'merge', f) for f in input_files]

    _main(output_path=out_folder, input_files=input_files, **parameter)


if __name__ == "__main__":
    main()
