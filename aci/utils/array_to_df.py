import pandas as pd
from .utils import int_to_string


def using_multiindex(A, columns, value_name='value'):
    shape = A.shape
    index = pd.MultiIndex.from_product([range(s) for s in shape], names=columns)
    df = pd.DataFrame({value_name: A.flatten()}, index=index).reset_index()
    return df


def map_columns(df, **map_columns):
    for col, names in map_columns.items():
        df[col] = df[col].map({idx: name for idx, name in enumerate(names)})
        # only make non numeric names to category
        try:
            int(names[0])
        except:
            df[col] = df[col].astype('category')
    return df


def to_alphabete(df, columns):
    # string_mapper = {x: y for x, y in enumerate(string.ascii_uppercase, 0)}
    for col in columns:
        df[col] = df[col].map(int_to_string)
    return df
