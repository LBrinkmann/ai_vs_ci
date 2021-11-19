import pandas as pd
import numpy as np
from functools import reduce
from itertools import product

comp = {
    "gt": lambda a, b: a > b,
    "lt": lambda a, b: a < b,
    "ge": lambda a, b: a >= b,
    "le": lambda a, b: a <= b,
}


def sub_filter(s, k, v):
    return comp[s](k, v)


def single_filter(s, f):
    if isinstance(f, list):
        w = s.isin(f)
    elif isinstance(f, dict):
        w = pd.Series(True, index=s.index)
        for k, v in f.items():
            w &= comp[k](s, v)
    else:
        w = s == f
    assert w.sum() > 0, f'Selecting for {f} does not leaves any data.'
    return w


def selector(df, selectors):
    if len(selectors) > 0:
        return reduce(
            lambda a, b: a & b,
            (single_filter(df[k], v) for k, v in selectors.items())
        )
    else:
        return pd.Series(True, index=df.index)


def select_df(df, selectors, remove=True):
    w = selector(df, selectors)

    if remove:
        remove_columns = [
            sel
            for sel, val in selectors.items()
            if not isinstance(val, list)
        ]
        columns = [
            c for c in df.columns if c not in remove_columns
        ]
    else:
        columns = df.columns
    return df.loc[w, columns]


def merge_selectors(selectors, new_selectors):
    """
    Merges two selectors. If the union is empty, return none.
    """
    for k, v in new_selectors.items():
        if k in selectors:
            if isinstance(selectors[k], list):
                if v not in selectors[k]:
                    return None
            elif selectors[k] != v:
                return None
    return {**selectors, **new_selectors}


def expand_selectors(df, selectors, expand=[]):
    if len(expand) == 0:
        yield selectors
    else:
        grid_dims = [[
            {e: f}
            for f in df[e].unique() if not pd.isnull(f)]
            for e in expand
        ]
        grid = list(product(*grid_dims))
        grid = list(map(merge_dicts, grid))
        for i, g in enumerate(grid):
            _selectors = merge_selectors(selectors, g)
            if _selectors is not None:
                yield _selectors


def combine_values(df, combine):
    w_keep = pd.Series(False, index=df.index)
    for k, v in combine.items():
        df[k] = np.nan
        for name, sel in v.items():
            w = selector(df, sel)
            w_keep |= w
            df.loc[w, k] = name

    remove_columns = set(name for v in combine.values()
                         for vv in v.values() for name in vv)
    columns = [c for c in df.columns if c not in remove_columns]
    return df.loc[w_keep, columns]


def merge_dicts(l):
    return reduce(lambda x, y: {**x, **y}, l, {})


def summarize_df(df, name):
    print(f'{name} has the columns: {", ".join(df.columns)}')
    for col in df.columns:
        if df[col].nunique() < 25:
            values = ', '.join(map(str, df[col].unique()))
            print(f'{name} {col} has the values: {values}')
        else:
            print(f'{name} {col} has {df[col].nunique()} unique values.')


def preprocess(df, combine=None, selectors={}):
    df = select_df(df, selectors).copy()
    if combine is not None:
        df = combine_values(df, combine)
    return df


def load_and_preprocess(filename, **preprocess_args):
    df = pd.read_parquet(filename)
    return preprocess(df, **preprocess_args)
