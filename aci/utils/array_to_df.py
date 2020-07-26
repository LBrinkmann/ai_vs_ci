import pandas as pd


def using_multiindex(A, columns):
    shape = A.shape
    index = pd.MultiIndex.from_product([range(s) for s in shape], names=columns)
    df = pd.DataFrame({'value': A.flatten()}, index=index).reset_index()
    return df


def map_columns(df, **map_columns):
    for col, names in map_columns.items():
        df[col] = df[col].map({idx: name for idx, name in enumerate(names)})
    return df


def to_alphabete(df, columns):
    import string
    string_mapper = {x: y for x, y in enumerate(string.ascii_uppercase, 0)}
    for col in columns:
        df[col] = df[col].map(string_mapper)
    return df