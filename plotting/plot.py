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
from multiprocessing import Manager, Pool
import matplotlib.pyplot as plt
import seaborn as sns

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
    return reduce(lambda a,b: a & b, (single_filter(df[k], v) for k, v in selectors.items()))

def plot(df, output_path, name, selectors, grid, hue=None, style=None):
    w = selector(df, selectors)

    title = ' | '.join(f"{k}:{v}" for k, v in selectors.items() if not isinstance(v, list))

    print(f'start plotting {name}')

    grid.sort(key=lambda l: df[l].nunique())

    grid = {n: g for g, n in zip(grid[::-1], ['col','row'])}

    g = sns.relplot(
        data=df[w], 
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

    filename = os.path.join(output_path, f"{name}.png")
    plt.savefig(filename)
    print(f'Saved {filename}')
    plt.close()


def preprocess(df, x, groupby, smooth=None, metrics=None, bins=None):
    groupby_reg = re.compile(groupby)
    groupby_columns = [c for c in df.columns if groupby_reg.search(c)]
    groupby_columns = [c for c in groupby_columns if df[c].nunique() > 1]

    if bins:
        df[f'{x}_bin'] = pd.cut(df[x], bins=bins).cat.codes
        groupby_columns.append(f'{x}_bin')
    print(groupby_columns)
    if smooth:
        df = df.groupby(groupby_columns).rolling(on=x, window=smooth)[metrics].mean().reset_index()
    else:
        df = df.groupby(groupby_columns + [x])[metrics].mean().reset_index()
    df = df.melt(id_vars=groupby_columns + [x], value_vars=metrics, var_name='metric')
    return df


def expand(df, plots):
    for p in plots:
        if 'expand' in p:
            for e in p['expand']:
                for f in df[e].unique():
                    _p = {k:v for k,v in p.items() if k != 'expand'}
                    yield {**_p, 'selectors': {**p['selectors'], e: f}, 'name': f"{p['name']}.{f}"}
        else:
            yield p


def main(input_file, preprocess_args, plot_args, output_path):
    df = pd.read_parquet(input_file.format(**os.environ))
    df = preprocess(df, **preprocess_args)
    plot_args = list(expand(df, plot_args))
    plot_args = [{**pa, 'output_path': output_path} for pa in plot_args]

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
