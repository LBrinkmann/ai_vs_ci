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

    return graph, graph_pos, data['agent_pos']


def make_video(arrays, episode, out_dir, mode):
    filename = os.path.join(out_dir, f"{mode}.{episode}.mp4")
    # array = np.stack(arrays, axis=0)
    clip = ImageSequenceClip(arrays, fps=1)
    print(filename)
    clip.write_videofile(filename)



def make_frame(df_step_actions, df_step_reward, out_dir, graph, graph_pos, agent_pos):
    # import ipdb; ipdb.set_trace()

    episode, episode_step, mode, _ = df_step_actions.index[0]

    fig = plt.figure()

    ci_reward = df_step_reward.loc[(episode, episode_step, mode, 'ci')]
    ai_reward = df_step_reward.loc[(episode, episode_step, mode, 'ci')]
    ci_action = df_step_actions.loc[(episode, episode_step, mode, 'ci')]
    ai_action = df_step_actions.loc[(episode, episode_step, mode, 'ci')]

    labels = {
        i: f"{ci:.2f}:{ai:.2f}" for i, (ci,ai) in enumerate(zip(ci_reward, ai_reward)) }
    
    # import ipdb; ipdb.set_trace()

    node_color_ci = [ci_action[i] for i in graph.nodes()]        
    node_color_ai = [ai_action[i] for i in graph.nodes()]
    nx.draw(graph, graph_pos, node_color=node_color_ai, node_size=500)

    nx.draw(
        graph, graph_pos, node_color=node_color_ci, 
        labels=labels, node_size=200, edgelist=[], font_size=20, font_color='magenta')

    plt.text(
        100, 100, f"{episode_step} CI:AI {sum(ci_reward)}:{sum(ai_reward)}", 
        fontsize=20, color='black'
    )
    fig.canvas.draw()
    plt.savefig('temp.png')
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # import ipdb; ipdb.set_trace()
    # data = np.moveaxis(data, 2, 0)
    plt.close()
    return data



def _main(out_dir, run_dir, period):
    action_files = find_files_with_match(run_dir, "actions.*parquet")
    reward_files = find_files_with_match(run_dir, "reward.*parquet")
    info_files = find_files_with_match(run_dir, "info.*parquet")
    df_actions = pd.concat(pd.read_parquet(p) for p in action_files)
    df_reward = pd.concat(pd.read_parquet(p) for p in reward_files)
    df_info = pd.concat(pd.read_parquet(p) for p in info_files)
    df_index = df_actions.index.to_frame()
    for mode, e_period in period.items():
        dfs = df_actions[(df_index['mode'] == mode) & (df_index['episode'] % e_period == 0)]
        for episode, df_eps_actions in dfs.groupby('episode'):
            frames = []
            graph, graph_pos, agent_pos = load_graph(run_dir, episode, mode)

            for episode_step, df_step_actions in df_eps_actions.groupby('episode_step'):
                df_step_reward = df_reward.loc[df_step_actions.index]
                frames.append(make_frame(df_step_actions, df_step_reward, out_dir, graph, graph_pos, agent_pos))
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