# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import seaborn as sns
import jsonlines
import matplotlib.pyplot as plt

def add_rolling_mean(df, columns, window, groupby, sortby):
    rolling = df.sort_values(sortby).groupby(groupby)[columns].rolling(window=window).mean()
    rolling.index = rolling.index.droplevel(level=0)
    rolling_columns = [f'{c}_rolling' for c in columns]
    df[rolling_columns] = rolling
    return df

# %%
repro = '~/cluster/repros/ai_vs_ci'
project = 'heuristic_tabular'
batch = 'tabular_vs_heuristic_2'


# %%
# df = pd.read_parquet('../runs/heuristic_tabular/heuristic_only/metrics.parquet')
final_df = pd.read_parquet(f'{repro}/runs/{project}/{batch}/agg/final.parquet')
trace_df = pd.read_parquet(f'{repro}/runs/{project}/{batch}/agg/trace.parquet')


# %%
final_df.columns


# %%
label_columns = [l for l in final_df.columns if 'label' in l]


metric_columns = ['ai_reward', 'sum_ci_reward', 'std_ci_reward', 'avg_coordination',
       'avg_catch']


# %%
# controller_args_order = ['settings1', 'settings2', 'settings3', 'settings4', 'settings5', 'settings6']
eps_order = ['fixed0.20', 'decay']
networktype_order = ['full3', 'cycle64']
rewards_order = ['individual',  'collective']
columns = metric_columns
window = 50
groupby = label_columns + ['mode']
sortby = 'episode'
groupby_unique = {
    l: final_df[l].unique().tolist()
    for l in groupby
}

# %%
final_smooth_df = final_df.groupby(groupby).rolling(on=sortby, window=window)[columns].mean().reset_index()

agg_trace_df = trace_df.groupby(label_columns + ['mode','episode_step'])[columns].mean().reset_index()
trace_df['episode_bin'] = pd.cut(trace_df['episode'], bins=10).cat.codes
binned_trace_df = trace_df.groupby(label_columns + ['mode', 'episode_step', 'episode_bin'])[columns].mean().reset_index()

final_smooth_dfm = final_smooth_df.melt(id_vars=label_columns + ['mode', 'episode'], value_vars=metric_columns)
agg_trace_dfm = agg_trace_df.melt(id_vars=label_columns + ['mode', 'episode_step'], value_vars=metric_columns)
binned_trace_dfm = binned_trace_df.melt(id_vars=label_columns + ['mode', 'episode_bin', 'episode_step'], value_vars=metric_columns)

# %% [Markdown]
## Coordination over episodes



# # %%
# rel_label_columns = [l for l in label_columns if len(groupby_unique[l]) > 1]

# for obs_column in rel_label_columns:
#     for i in range(6):
#         import random
#         from functools import reduce

#         sel_label_columns = [l for l in rel_label_columns if not l == obs_column]

        
#         grid_label = random.sample(sel_label_columns, 2)
#         other_label =  [l for l in sel_label_columns if (l not in grid_label)]
#         fixed_values = {
#             l: random.choice(groupby_unique[l])
#             for l in other_label
#         }

#         w1 = reduce(lambda a,b: a & b, (final_smooth_dfm[k] == v for k,v in fixed_values.items()))
#         w2 = final_smooth_dfm['variable'].isin(['avg_coordination'])
#         w = w1 & w2

#         title = ' | '.join(f"{k}:{v}" for k, v in fixed_values.items())

#         grid_label.sort(key=lambda l: len(groupby_unique[l]))

#         g = sns.relplot(
#             data=final_smooth_dfm[w], 
#             x='episode', 
#             y='value', 
#             row=grid_label[0], 
#             col=grid_label[1],  
#             hue=obs_column, 
#             kind="line", 
#             style='mode', 
#             ci=None
#         )
#         plt.subplots_adjust(top=0.9)
#         g.fig.suptitle(title)
#         g.fig.patch.set_facecolor('white')
#         # g.set_title('Title')

#         plt.savefig(f"plots/tabular_vs_heuristic_2/{obs_column}_{i}.png")
#         plt.close()



# # %%
# rel_label_columns = [l for l in label_columns if len(groupby_unique[l]) > 1]

# for obs_column in rel_label_columns:
#     import random
#     from functools import reduce

#     sel_label_columns = [l for l in rel_label_columns if not l == obs_column]

    
#     grid_label = random.sample(sel_label_columns, 2)
#     other_label =  [l for l in sel_label_columns if (l not in grid_label)]
#     # fixed_values = {
#     #     l: random.choice(groupby_unique[l])
#     #     for l in other_label
#     # }

#     # w1 = reduce(lambda a,b: a & b, (final_smooth_dfm[k] == v for k,v in fixed_values.items()))
#     w2 = final_smooth_dfm['variable'].isin(['avg_coordination'])
#     w = w2

#     # title = ' | '.join(f"{k}:{v}" for k, v in other_label)

#     grid_label.sort(key=lambda l: len(groupby_unique[l]))

#     g = sns.relplot(
#         data=final_smooth_dfm[w], 
#         x='episode', 
#         y='value', 
#         row='mode', 
#         col=obs_column,  
#         kind="line"
#     )
#     plt.subplots_adjust(top=0.9)
#     # g.fig.suptitle(title)
#     g.fig.patch.set_facecolor('white')

#     plt.savefig(f"plots/tabular_vs_heuristic_2/2#{obs_column}.png")
#     plt.close()

# # %%
# fixed_values = {
#     'label.gamma': 0.9,
#     'label.cache_size': 1,
# }

# rel_label_columns = [l for l in label_columns if (len(groupby_unique[l]) > 1) and l not in fixed_values]

# for obs_column in rel_label_columns:
#     import random
#     from functools import reduce

#     sel_label_columns = [l for l in rel_label_columns if not l == obs_column]

    
#     grid_label = random.sample(sel_label_columns, 2)
#     other_label =  [l for l in sel_label_columns if (l not in grid_label)]


#     w1 = reduce(lambda a,b: a & b, (final_smooth_dfm[k] == v for k,v in fixed_values.items()))
#     w2 = final_smooth_dfm['variable'].isin(['avg_coordination'])
#     w = w2

#     title = ' | '.join(f"{k}:{v}" for k, v in fixed_values.items())

#     grid_label.sort(key=lambda l: len(groupby_unique[l]))

#     g = sns.relplot(
#         data=final_smooth_dfm[w], 
#         x='episode', 
#         y='value', 
#         row='mode', 
#         col=obs_column,  
#         kind="line"
#     )
#     plt.subplots_adjust(top=0.9)
#     # g.fig.suptitle(title)
#     g.fig.patch.set_facecolor('white')

#     plt.savefig(f"plots/tabular_vs_heuristic_2/3#{obs_column}.png")
#     plt.close()


# %%

import random
from functools import reduce

fixed_values = {
    # 'label.gamma': 0.9,
    'label.eps': 'fixed0.20',
    'label.cache_size': 1,
    'label.q_start': 0, 
    'label.rewards': 'individual'
}
side_effects = ['label.alpha','label.gamma',]

obs_column = 'label.networktype'

w1 = reduce(lambda a,b: a & b, (final_smooth_dfm[k] == v for k,v in fixed_values.items()))
w2 = final_smooth_dfm['variable'].isin(['avg_coordination'])
w = w1 & w2

title = ' | '.join(f"{k}:{v}" for k, v in fixed_values.items())

print(f'start plotting {len(final_smooth_dfm[w])}')


g = sns.relplot(
    data=final_smooth_dfm[w], 
    x='episode', 
    y='value', 
    row=side_effects[0], 
    col=side_effects[1],
    hue=obs_column,
    style='mode',
    kind="line",
    ci=None
)
plt.subplots_adjust(top=0.9)
g.fig.suptitle(title)
g.fig.patch.set_facecolor('white')

filename = f"plots/tabular_vs_heuristic_2/0.gamma.png"
print(filename)
plt.savefig(filename)
plt.close()



# %%

import random
from functools import reduce

fixed_values = {
    'label.gamma': 0.9,
    # 'label.cache_size': 1,
    # 'label.q_start': 0, 
    # 'label.rewards': 'individual',
    'label.alpha': 0.1,
    'label.networktype': 'cycle64',
    'label.eps': 'fixed0.20'
}
side_effects = ['label.rewards','label.cache_size',]

obs_column = 'label.q_start'

w1 = reduce(lambda a,b: a & b, (final_smooth_dfm[k] == v for k,v in fixed_values.items()))
w2 = final_smooth_dfm['variable'].isin(['avg_coordination'])
w = w1 & w2

title = ' | '.join(f"{k}:{v}" for k, v in fixed_values.items())

print(f'start plotting {len(final_smooth_dfm[w])}')


g = sns.relplot(
    data=final_smooth_dfm[w], 
    x='episode', 
    y='value', 
    row=side_effects[0], 
    col=side_effects[1],
    hue=obs_column,
    style='mode',
    kind="line",
    ci=None
)
plt.subplots_adjust(top=0.9)
g.fig.suptitle(title)
g.fig.patch.set_facecolor('white')

filename = f"plots/tabular_vs_heuristic_2/1.{obs_column}.png"
print(filename)
plt.savefig(filename)
plt.close()


# %%

import random
from functools import reduce

fixed_values = {
    'label.gamma': 0.9,
    # 'label.cache_size': 1,
    # 'label.q_start': 0, 
    # 'label.rewards': 'individual',
    'label.alpha': 0.1,
    'label.networktype': 'cycle64',
    'label.eps': 'fixed0.20'
}
side_effects = ['label.rewards','label.cache_size',]

obs_column = 'label.q_start'

w1 = reduce(lambda a,b: a & b, (final_smooth_dfm[k] == v for k,v in fixed_values.items()))
w2 = final_smooth_dfm['variable'].isin(['avg_coordination'])
w = w1 & w2

title = ' | '.join(f"{k}:{v}" for k, v in fixed_values.items())

print(f'start plotting {len(final_smooth_dfm[w])}')


g = sns.relplot(
    data=final_smooth_dfm[w], 
    x='episode', 
    y='value', 
    row=side_effects[0], 
    col=side_effects[1],
    hue=obs_column,
    style='mode',
    kind="line",
    ci=None
)
plt.subplots_adjust(top=0.9)
g.fig.suptitle(title)
g.fig.patch.set_facecolor('white')

filename = f"plots/tabular_vs_heuristic_2/1.{obs_column}.png"
print(filename)
plt.savefig(filename)
plt.close()


# %%
