"""Usage: train.py RUN_FOLDER

Arguments:
    RUN_FOLDER

Outputs:
    ....
"""

from docopt import docopt
import os
import pandas as pd
import numpy as np
import multiprocessing
import itertools
import re
import json
import networkx as nx
import matplotlib.pyplot as plt
from aci.utils.io import load_yaml
from moviepy.editor import ImageSequenceClip


N_JOBS = 16


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def read_file(args):
    filename, fullfilename = args
    return (filename, pd.read_parquet(fullfilename))


def find_files_with_match(_dir, patter):
    for root, dirs, files in os.walk(_dir):
        for file in files:
            if re.search(patter, file):
                yield os.path.join(root, file)


def load_graph(run_dir, episode, mode):
    filename = os.path.join(run_dir, 'train', 'envs', f"{mode}.{episode}.json")
    with open(filename) as json_file:
        data = json.load(json_file)

    graph = nx.from_edgelist(data['graph']) 
    graph_pos = nx.spring_layout(graph)

    return graph, graph_pos


def make_video(arrays, episode, out_dir, mode):
    filename = os.path.join(out_dir, f"{mode}.{episode}.mp4")
    # array = np.stack(arrays, axis=0)
    clip = ImageSequenceClip(arrays, fps=1)
    print(filename)
    clip.write_videofile(filename)



def make_frame(df_step_actions, df_step_info, out_dir, graph, graph_pos):
    # import ipdb; ipdb.set_trace()

    episode, episode_step, mode, _ = df_step_actions.index[0]

    fig = plt.figure()

    ci_action = df_step_actions.loc[(episode, episode_step, mode, 'ci')]
    ai_action = df_step_actions.loc[(episode, episode_step, mode, 'ai')]

    node_color_ci = [ci_action[i] for i in graph.nodes()]        
    node_color_ai = ['black' if ai_action[i] == ci_action[i] else 'white' for i in graph.nodes()]

    is_coordination = df_step_info.loc['ind_coordination'].values
    coordination_color = ['green' if is_coordination[i] == 1 else 'red' for i in graph.nodes()]

    nx.draw(graph, graph_pos, node_color=node_color_ai, node_size=500)

    nx.draw(graph, graph_pos, node_color=node_color_ci, node_size=300, edgelist=[])

    nx.draw(graph, graph_pos, node_color=coordination_color, node_size=30, edgelist=[])

    coordination = df_step_info.loc['avg_coordination'].sum()
    catches = df_step_info.loc['avg_catch'].sum()

    plt.text(
        100, 100, f"{episode_step} CI:AI {coordination}:{catches}", 
        fontsize=20, color='black'
    )
    fig.canvas.draw()
    plt.savefig(f'tmp/{mode}.{episode}.{episode_step}.png')
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # import ipdb; ipdb.set_trace()
    # data = np.moveaxis(data, 2, 0)
    plt.close()
    return data



def _main(out_dir, run_dir, period):
    action_files = find_files_with_match(run_dir, "actions.*parquet")
    reward_files = find_files_with_match(run_dir, "^rewards.*parquet")
    info_files = find_files_with_match(run_dir, "^info.*parquet")
    df_actions = pd.concat(pd.read_parquet(p) for p in action_files)
    df_reward = pd.concat(pd.read_parquet(p) for p in reward_files)
    df_info = pd.concat(pd.read_parquet(p) for p in info_files)

    df_index = df_actions.index.to_frame()
    for mode, e_period in period.items():
        dfs = df_actions[(df_index['mode'] == mode) & (df_index['episode'] % e_period == 0)]
        for episode, df_eps_actions in dfs.groupby('episode'):
            print(f'Start with episode {episode} and mode {mode}.')
            frames = []
            graph, graph_pos = load_graph(run_dir, episode, mode)

            for episode_step, df_step_actions in df_eps_actions.groupby('episode_step'):
                df_step_info = df_info.loc[df_step_actions.iloc[0].name[0:3]]
                # coordination = df_step_info.sum(1)['avg_coordination']
                # catch = df_step_info.sum(1)['avg_catch']
                # ind_coordination = df_step_info.loc['avg_coordination'].values
                # info = {'coordination': coordination, 'catch': catch, 'ind_coordination': ind_coordination}
                frames.append(make_frame(df_step_actions, df_step_info, out_dir, graph, graph_pos))
            make_video(frames, episode=episode, mode=mode, out_dir=out_dir)


def main():
    arguments = docopt(__doc__)
    run_dir = arguments['RUN_FOLDER']

    parameter_file = os.path.join(run_dir, 'video.yml')
    out_dir = os.path.join(run_dir, 'video')
    parameter = load_yaml(parameter_file)
    ensure_dir(out_dir)

    _main(run_dir=run_dir, out_dir=out_dir, **parameter)



if __name__ == "__main__":
    main()