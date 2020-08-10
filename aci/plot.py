"""Usage: train.py PARAMETER_FILE OUTPUT_PATH

Arguments:
    RUN_PATH

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

from aci.utils.io import load_yaml


def _plot(args):
    plot(args[1].df, **args[0])

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

def plot(df, output_path, name, selectors, grid=[], hue=None, style=None, folder=None):
    w = selector(df, selectors)

    title = ' | '.join(f"{k}:{v}" for k, v in selectors.items() if not isinstance(v, list))

    print(f'start plotting {name}')

    dfs = df[w].copy()
    if pd.api.types.is_numeric_dtype(dfs[hue].dtype):
        dfs[hue] = "#" + dfs[hue].astype(str)

    grid.sort(key=lambda l: dfs[l].nunique())

    grid = {n: g for g, n in zip(grid[::-1], ['col','row'])}

    g = sns.relplot(
        data=dfs, 
        x='episode', 
        y='value', 
        **grid,
        hue=hue,
        style=style,
        kind="line",
        ci=None
    )
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(title)
    g.fig.patch.set_facecolor('white')
    if folder: 
        ensure_directory(os.path.join(output_path, folder))
        filename = os.path.join(output_path, folder, f"{name}.png")
    else:
        filename = os.path.join(output_path, f"{name}.png")
    plt.savefig(filename)
    print(f'Saved {filename}')
    plt.close()


def preprocess(df, x, groupby, smooth=None, metrics=None, bins=None):

    print(df.head())

    groupby_reg = re.compile(groupby)
    groupby_columns = [c for c in df.columns if groupby_reg.search(c)]
    groupby_columns = [c for c in groupby_columns if df[c].nunique() > 1]

    if bins:
        df[f'{x}_bin'] = pd.cut(df[x], bins=bins).cat.codes
        groupby_columns.append(f'{x}_bin')
    
    for gc in groupby_columns:
        values = ', '.join(map(str, df[gc].unique()))
        print(f'{gc} has the values: {values}')

    print(', '.join(groupby_columns))
    if smooth:
        assert df.groupby(groupby_columns, observed=True)[x].count().min() > smooth, 'To few datapoints for smoothing.'
        df = df.groupby(groupby_columns, observed=True).rolling(on=x, window=smooth)[metrics].mean().reset_index()
    else:
        df = df.groupby(groupby_columns + [x], observed=True)[metrics].mean().reset_index()
    df = df.melt(id_vars=groupby_columns + [x], value_vars=metrics, var_name='metric')
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
            grid_dims = [[{e: f} for f in df[e].unique()] for e in p['expand']]
            grid = list(product(*grid_dims))
            grid = list(map(merge_dicts, grid))
            _p = {k:v for k,v in p.items() if k != 'expand'}
            for i, g in enumerate(grid):
                if check_selector(p['selectors'], g):
                    yield {**_p, 'selectors': {**p['selectors'], **g}, 'name': str(i), 'folder': p['name']}
        else:
            yield p


def main(input_file, clean, preprocess_args, plot_args, output_path):
    if clean:
        shutil.rmtree(output_path, ignore_errors=True)
    ensure_directory(output_path)
    df = pd.read_parquet(input_file.format(**os.environ))
    df = preprocess(df, **preprocess_args)
    plot_args = list(expand(df, plot_args))
    plot_args = [{**pa, 'output_path': output_path} for pa in plot_args]

    print(f"Make {len(plot_args)} plots.")

    mgr = Manager()
    ns = mgr.Namespace()
    ns.df = df
    pool = Pool(20)
    data_args = list(zip(plot_args, [ns]*len(plot_args)))

    # print(data_args)

    pool.map(_plot, data_args)

    # for p in plot_args:
    #     plot(df, **p)


if __name__ == "__main__":
    arguments = docopt(__doc__)
    parameter_file = arguments['PARAMETER_FILE']
    output_path = arguments['OUTPUT_PATH']
    parameter = load_yaml(parameter_file)

    main(output_path=output_path, **parameter)
