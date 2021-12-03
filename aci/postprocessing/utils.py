import pandas as pd


def aggregation(df, quarter_column, quarter_name, bin_name, bin_column, bin_size, agg_column=None, agg_func='count'):
    columns = list(set(df.columns) - set((quarter_column, bin_column, agg_column))) + [bin_name]

    # create column to aggregate first and last quarter seperatly
    max_episode_step = df[quarter_column].max()
    df[quarter_name] = 'middle'
    df.loc[df[quarter_column] < max_episode_step * 0.25, quarter_name] = 'start'
    df.loc[df[quarter_column] > max_episode_step * 0.75, quarter_name] = 'end'

    # create column to bin episodes
    df[bin_name] = df[bin_column] // bin_size * bin_size + bin_size // 2
    if agg_column is None:
        df['count'] = 1
        agg_column = 'count'

    # aggreation with quarters seperatly
    df_agg = df.groupby(columns + [quarter_name])[agg_column].agg(agg_func)
    df_agg = df_agg.reset_index()

    # aggreation with full data
    df_all_agg = df.groupby(columns)[agg_column].agg(agg_func)
    df_all_agg = df_all_agg.reset_index()
    df_all_agg[quarter_name] = 'full'

    df = pd.concat([df_agg, df_all_agg])
    return df


def add_all(df, value_name, merge_column, sum_name, agg_func):
    groupby_columns = list(set(df.columns) - set([value_name, merge_column]))
    sum_df = df.groupby(groupby_columns)[value_name].agg(agg_func).reset_index()
    sum_df[merge_column] = sum_name
    return pd.concat([df, sum_df])
