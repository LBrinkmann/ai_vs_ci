import pandas as pd
import numpy as np
from itertools import product


# create pattern df

def de_duplicate(seq):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def normalize_pattern(pattern, actions):
    """
    normalize by mapping permutations of actions
    """
    mapping = {name: actions[idx]
               for idx, name in enumerate(de_duplicate(pattern))}
    return ''.join(mapping[p] for p in pattern)


def normalize_pattern_shifts(pattern):
    """
    normalize by mapping shifts of the pattern
    """
    shifted_pattern = [
        pattern[i:] + pattern[:i]
        for i in range(len(pattern))
    ]

    return sorted(shifted_pattern)[0]


def create_single_pattern_df(length, actions):
    pattern_name = list(''.join(n) for n in product(actions, repeat=length))
    df = pd.DataFrame(
        {
            'action_ids': list(product(range(len(actions)), repeat=length)),
            'pattern_name': pattern_name,
            'pattern_length': length,
        },
    )
    return df


def create_pattern_df(min_length, max_length, actions):
    """
    Create tuples from the outer product of avaible actions.
    Tuples which are equivalent upon shifts are
    assign the same pattern group.

    Parameters
    ----------
    min_length : int
        minimum length of tuple
    max_length : int
        maximum length of tuple
    actions : list
        name of actions

    Returns
    -------
    pd.DataFrame
        Dataframe with patterns. Columns are as follows:
        ==================  ==========================================
        pattern_id          Id of pattern.
        pattern_length      Length of pattern.
        pattern_name        Name of pattern as joined as action names.
        pattern_group_name  Pattern group.
        action_ids          Tuple with action ids of the pattern.
        ==================  ==========================================

    """
    pattern_dfs = [
        create_single_pattern_df(length, actions)
        for length in range(min_length, max_length + 1)
    ]
    pattern_df = pd.concat(pattern_dfs)
    pattern_df['pattern_id'] = np.arange(len(pattern_df))
    pattern_df['pattern_group_name'] = pattern_df['pattern_name'].map(
        lambda p: normalize_pattern_shifts(p))
    category_columns = ['pattern_name', 'pattern_group_name']
    pattern_df[category_columns] = pattern_df[category_columns].astype(
        'category')
    column_order = [
        'pattern_id', 'pattern_length', 'pattern_name',
        'pattern_group_name', 'action_ids'
    ]

    return pattern_df[column_order]
