"""Usage: train.py RUN_FOLDER

Arguments:
    RUN_FOLDER

Outputs:
    ....
"""

from docopt import docopt
import pandas as pd
import os
import re
from functools import reduce
from itertools import product
from multiprocessing import Manager, Pool
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import numpy as np

from aci.utils.io import load_yaml, ensure_dir


def _plot(args):
    plot(args[1].dfs, **args[0])

def single_filter(s, f):
    if isinstance(f, list):
        w = s.isin(f)
    else:
        w = s == f
    assert w.sum() > 0, f'Selecting for {f} does not leaves any data.'
    return w


def selector(df, selectors):
    if len(selectors) > 0:
        return reduce(lambda a,b: a & b, (single_filter(df[k], v) for k, v in selectors.items()))
    else:
        return pd.Series(True, index=df.index)

def plot(
    dfs, output_path, filename, name, selectors, x, y, grid=[], baseline=None,
    hue=None, style=None, folder=None,):
    df = dfs[filename]

    w = selector(df, selectors)
    # wb = selector(df, {**selectors, **baseline})

    title = ' | '.join(f"{k}:{v}" for k, v in selectors.items() if not isinstance(v, list))

    print(f'start plotting {name}')

    dfs = df[w].copy()
    # dfsb = df[wb].copy()

    if (hue is not None) and pd.api.types.is_numeric_dtype(dfs[hue].dtype):
        dfs[hue] = "#" + dfs[hue].astype(str)

    for col, iscat in (df.dtypes == 'category').iteritems():
        if iscat:
            dfs[col] = dfs[col].cat.remove_unused_categories()
            # dfsb[col] = dfsb[col].cat.remove_unused_categories()

    # assert indiviuallity
    groupby = [c for c in (grid + [hue, style, x]) if c is not None]

    dfs = dfs.dropna(subset=groupby)
    # dfsb = dfsb.dropna(subset=groupby)

    w_baseline = dfs[baseline['column']] == baseline['value']
    df_dup = dfs[~w_baseline].duplicated(subset=groupby)
    if df_dup.any():
        dup = dfs[df_dup].sort_values(groupby)
        print(dup.head())
        diff = dup.iloc[0] != dup.iloc[1]
        print(dup.loc[:,diff])
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
    if folder:
        ensure_directory(os.path.join(output_path, folder))
        filename = os.path.join(output_path, folder, f"{name}.png")
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





def preprocess(df, combine={}, selectors={}):
    w = selector(df, selectors)
    df = df[w].copy()
    for k, v in combine.items():
        df[k] = np.nan
        for name, sel in v.items():
            w = selector(df, sel)
            df.loc[w, k] = name
    return df


def merge_dicts(l):
    return reduce(lambda x,y: {**x, **y}, l, {})


def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def check_selector(selectors, new_selectors):
    for k, v in new_selectors.items():
        if k in selectors:
            if isinstance(selectors[k], list):
                if v not in selectors[k]:
                    return False
            elif selectors[k] != v:
                return False
    return True


def expand(df, plots):
    for p in plots:
        if 'expand' in p:
            grid_dims = [[{e: f} for f in df[e].unique() if not pd.isnull(f)] for e in p['expand']]
            grid = list(product(*grid_dims))
            grid = list(map(merge_dicts, grid))
            _p = {k:v for k,v in p.items() if k != 'expand'}
            for i, g in enumerate(grid):
                if check_selector(p['selectors'], g):
                    yield {**_p, 'selectors': {**p['selectors'], **g}, 'name': str(i), 'folder': p['name']}
        else:
            yield p


def get_subfolders(d):
    return [o for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]


def _main(*, input_files, clean, preprocess_args=None, plot_args, output_path):
    if clean:
        shutil.rmtree(output_path, ignore_errors=True)
    ensure_directory(output_path)

    input_files = {
        os.path.split(f)[-1].split('.')[-2]: f
        for f in input_files
    }

    # dfs = {
    #     os.path.split(f)[-1].split('.')[-2]: pd.read_parquet(f)
    #     for f,  in input_files
    # }

    dfs = {
        name: preprocess(pd.read_parquet(input_files[name]), **pa)
        for name, pa in preprocess_args.items()
    }


    for name, df in dfs.items():
        print(f'{name} has the columns: {", ".join(df.columns)}')
        for col in df.columns:
            if df[col].nunique() < 25:
                values = ', '.join(map(str, df[col].unique()))
                print(f'{name} {col} has the values: {values}')

    plot_args = list(expand(df, plot_args))
    plot_args = [{**pa, 'output_path': output_path} for pa in plot_args]

    print(f"Make {len(plot_args)} plots.")


    mgr = Manager()
    ns = mgr.Namespace()
    ns.dfs = dfs
    pool = Pool(20)
    data_args = list(zip(plot_args, [ns]*len(plot_args)))
    # pool.map(_plot, data_args)
    _plot(data_args[0])
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
