# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import seaborn as sns
import jsonlines


# %%
def add_rolling_mean(df, columns, window, groupby, sortby):
    rolling = df.sort_values(sortby).groupby(groupby)[columns].rolling(window=window).mean()
    rolling.index = rolling.index.droplevel(level=0)
    rolling_columns = [f'{c}_rolling' for c in columns]
    df[rolling_columns] = rolling
    return df


# %%
# df = pd.read_parquet('../runs/heuristic_tabular/heuristic_only/metrics.parquet')
final_df = pd.read_parquet(
    '~/cluster/repros/ai_vs_ci/runs/heuristic_tabular/heuristic_only/agg/final.parquet')


# %%
final_df.head(2)


# %%
len(final_df)


# %%
final_df.columns


# %%
label_columns = ['label.batch', 'label.eps', 'label.networktype', 'label.project',
       'label.self_weight']

metric_columns = ['ai_reward', 'sum_ci_reward', 'std_ci_reward', 'avg_coordination',
       'avg_catch']


# %%
print(final_df['label.eps'].unique())
print(final_df['label.networktype'].unique())


# %%
eps_order = ['fixed0.05', 'fixed0.10', 'fixed0.20', 'decay']
networktype_order = ['full2', 'full3', 'full5', 'cycle62', 'cycle64', 'reg10_4_4']
columns = metric_columns
window = 50
groupby = label_columns + ['mode']
sortby = 'episode'


# %%


final_smooth_df = final_df.groupby(groupby).rolling(on=sortby, window=window)[columns].mean().reset_index()


# %%
trace_df = pd.read_parquet(
    '~/cluster/repros/ai_vs_ci/runs/heuristic_tabular/heuristic_only/agg/trace.parquet')

agg_trace_df = trace_df.groupby(label_columns + ['mode', 'episode_step'])[columns].mean().reset_index()


trace_df['episode_bin'] = pd.cut(trace_df['episode'], bins=10).cat.codes

binned_trace_df = trace_df.groupby(label_columns + ['mode', 'episode_step', 'episode_bin'])[columns].mean().reset_index()


# %%
sns.relplot(
    data=final_smooth_df, x='episode', y='avg_coordination', row='label.eps', col="label.networktype",  
    hue="label.self_weight", kind="line", row_order=eps_order, col_order=networktype_order, ci=None)


# %%
sns.relplot(
    data=agg_trace_df, x='episode_step', y='avg_coordination', row='label.eps', style='mode', col="label.networktype", hue="label.self_weight", kind="line", row_order=eps_order, col_order=networktype_order, ci=None)


# %%
sns.relplot(
    data=agg_trace_df, x='episode_step', y='avg_catch', row='label.eps', style='mode', col="label.networktype", hue="label.self_weight", kind="line", row_order=eps_order, col_order=networktype_order, ci=None)


# %%
sns.relplot(
    data=agg_trace_df, x='episode_step', y='sum_ci_reward', row='label.eps', style='mode', col="label.networktype", hue="label.self_weight", kind="line", row_order=eps_order, col_order=networktype_order, ci=None)


# %%
w_selection = (binned_trace_df['label.self_weight'] == 1) & (binned_trace_df['mode'] == 'train')

sns.relplot(
    data=binned_trace_df[w_selection], x='episode_step', y='avg_coordination', row='label.eps', style='mode', col="label.networktype", hue="episode_bin", kind="line", row_order=eps_order, col_order=networktype_order, ci=None)

# %% [markdown]
# to be done: individual agent rewards
