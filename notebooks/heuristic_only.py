# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import seaborn as sns
import jsonlines

def add_rolling_mean(df, columns, window, groupby, sortby):
    rolling = df.sort_values(sortby).groupby(groupby)[columns].rolling(window=window).mean()
    rolling.index = rolling.index.droplevel(level=0)
    rolling_columns = [f'{c}_rolling' for c in columns]
    df[rolling_columns] = rolling
    return df

# %%
repro = '~/cluster/repros/ai_vs_ci'
project = 'heuristic_tabular'
batch = 'heuristic_only'


# %%
# df = pd.read_parquet('../runs/heuristic_tabular/heuristic_only/metrics.parquet')
final_df = pd.read_parquet(f'{repro}/runs/{project}/{batch}/agg/final.parquet')
trace_df = pd.read_parquet(f'{repro}/runs/{project}/{batch}/agg/trace.parquet')


# %%
final_df.columns


# %%
label_columns = [l for l in final_df.columns if 'label' in l]
for l in label_columns:
    print(l, final_df[l].unique())

metric_columns = ['ai_reward', 'sum_ci_reward', 'std_ci_reward', 'avg_coordination',
       'avg_catch']


# %%
eps_order = ['fixed0.05', 'fixed0.10', 'fixed0.20', 'decay']
networktype_order = ['full2', 'full3', 'full5', 'cycle62', 'cycle64', 'reg10_4_4']
columns = metric_columns
window = 50
groupby = label_columns + ['mode']
sortby = 'episode'


# %%
final_smooth_df = final_df.groupby(groupby).rolling(on=sortby, window=window)[columns].mean().reset_index()

agg_trace_df = trace_df.groupby(label_columns + ['mode', 'episode_step'])[columns].mean().reset_index()
trace_df['episode_bin'] = pd.cut(trace_df['episode'], bins=10).cat.codes
binned_trace_df = trace_df.groupby(label_columns + ['mode', 'episode_step', 'episode_bin'])[columns].mean().reset_index()

final_smooth_dfm = final_smooth_df.melt(id_vars=label_columns + ['mode', 'episode'], value_vars=metric_columns)
agg_trace_dfm = agg_trace_df.melt(id_vars=label_columns + ['mode', 'episode_step'], value_vars=metric_columns)
binned_trace_dfm = binned_trace_df.melt(id_vars=label_columns + ['mode', 'episode_bin', 'episode_step'], value_vars=metric_columns)

# %% [Markdown]
## Coordination over episodes

# %%
w = (
    (final_smooth_dfm['mode'] == 'train') &
    final_smooth_dfm['variable'].isin(['avg_catch' , 'avg_coordination'])
)

sns.relplot(
    data=final_smooth_dfm[w], 
    x='episode', 
    y='value', 
    row='label.eps', 
    col="label.networktype",  
    hue="label.self_weight", 
    kind="line", 
    row_order=eps_order, 
    col_order=networktype_order,
    style='variable',
    ci=None
)


# %%

w = (
    (agg_trace_dfm['mode'] == 'train') &
    agg_trace_dfm['variable'].isin(['avg_catch' , 'avg_coordination'])
)


sns.relplot(
    data=agg_trace_dfm[w], 
    x='episode_step', 
    y='value', 
    row='label.eps', 
    style='variable', 
    col="label.networktype", 
    hue="label.self_weight", 
    kind="line", 
    row_order=eps_order, 
    col_order=networktype_order, 
    ci=None
)

# %%

w = (
    (binned_trace_dfm['label.self_weight'] == 1) &
    (binned_trace_dfm['mode'] == 'train') &
    binned_trace_dfm['variable'].isin(['avg_catch' , 'avg_coordination'])
)


sns.relplot(
    data=binned_trace_dfm[w], 
    x='episode_step', 
    y='value', 
    row='label.eps', 
    style='variable', 
    col="label.networktype", 
    hue="episode_bin", 
    kind="line", 
    row_order=eps_order, 
    col_order=networktype_order, 
    ci=None
)



# %%
