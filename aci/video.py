"""Usage: train.py RUN_FOLDER [IN_FOLDER]

Arguments:
    RUN_FOLDER

Outputs:
    ....
"""

from docopt import docopt
import os
import numpy as np
import torch as th
import networkx as nx
import matplotlib.pyplot as plt
from aci.utils.io import load_yaml, ensure_dir
from moviepy.editor import ImageSequenceClip
from aci.post import get_files
from multiprocessing import Manager, Pool


colormap = {
    0: 'black',
    1: 'green',
    -1: 'red'
}


def make_frame(
        actions, reward, metrics, fullmetrics, agent_types, metric_names, agents, step, graph, graph_pos, mode,
        episode, **_):
    fig = plt.figure()

    action_0 = actions[:, 0]
    action_1 = actions[:, 1]

    node_color_0 = [action_0[i].item() for i in graph.nodes()]
    node_color_1 = [action_1[i].item() for i in graph.nodes()]

    anticoor = metrics[:, 0, metric_names.index('ind_anticoordination')]
    crosscoor = metrics[:, 1, metric_names.index('ind_crosscoordination')]
    coordination_color = [colormap[int(anticoor[i])] for i in graph.nodes()]
    catch_color = [colormap[int(crosscoor[i])] for i in graph.nodes()]

    nx.draw(graph, graph_pos, node_color=catch_color, node_size=700)

    nx.draw(graph, graph_pos, node_color=node_color_1, node_size=550)

    nx.draw(graph, graph_pos, node_color=node_color_0,
            node_size=300, edgelist=[])

    nx.draw(graph, graph_pos, node_color=coordination_color,
            node_size=30, edgelist=[])

    reward_0 = reward[:, 0].sum()
    reward_1 = reward[:, 1].sum()

    print(fullmetrics.shape)

    anticoor_cum = fullmetrics[:, :, 0, metric_names.index(
        'ind_anticoordination')].sum(1).mean()
    crosscoor_cum = fullmetrics[:, :, 0, metric_names.index(
        'ind_crosscoordination')].sum(1).mean()

    crosscoor_m = crosscoor.sum()
    anticoor_m = anticoor.sum()

    text = f"{step} agents {agent_types[0]}:{agent_types[1]}" \
        f" reward {reward_0:.1f}:{reward_1:.1f}" \
        f" anticoor {anticoor_m} ({anticoor_cum:.1f}) catch {crosscoor_m} ({crosscoor_cum:.1f})"

    min_xpos = min(p[0] for p in graph_pos.values())
    min_ypos = min(p[1] for p in graph_pos.values())
    plt.text(min_xpos, min_ypos, text, fontsize=10, color='black',
             horizontalalignment='left', verticalalignment='top')
    fig.canvas.draw()
    # plt.savefig(f'tmp/{mode}.{episode}.{step}.png')
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # import ipdb; ipdb.set_trace()
    # data = np.moveaxis(data, 2, 0)
    plt.close()
    plt.clf()
    return data


def get_graph(neighbors):
    edgelist = [[i, j.item()] for i, n in enumerate(neighbors[:, 1:])
                for j in n if j != -1]
    graph = nx.from_edgelist(edgelist)
    graph_pos = nx.spring_layout(graph)
    return graph, graph_pos


def make_video(episode, neighbors, actions, reward, metrics, outdir, mode, **others):
    filename = os.path.join(outdir, f"{mode}.{episode}.mp4")
    graph, graph_pos = get_graph(neighbors)
    array = [
        make_frame(
            actions=a, reward=r, metrics=m, fullmetrics=metrics[:i+1],
            graph=graph, step=i, graph_pos=graph_pos, mode=mode,
            episode=episode, **others)
        for i, (a, r, m) in enumerate(zip(actions, reward, metrics))
    ]
    clip = ImageSequenceClip(array, fps=1)
    clip.write_videofile(filename, codec='mpeg4')


def select_episodes(episode, mod_episode, **tensor):
    for i, e in enumerate(episode):
        if (e % mod_episode) == 0:
            copy = ['agent_types', 'agents', 'metric_names']
            select = ['neighbors', 'actions', 'reward', 'metrics', 'agent_map']
            yield {
                'episode': e,
                **{k: tensor[k] for k in copy},
                **{k: tensor[k][i] for k in select}
            }


def _make_videos(
        filename, mode, labels, outdir, name, mod_episode):
    tensors = th.load(filename, map_location=th.device('cpu'))
    ensure_dir(outdir)

    for selected in select_episodes(**tensors,  mod_episode=mod_episode):
        make_video(**selected, labels=labels, mode=mode, outdir=outdir)


def make_videos(args):
    return _make_videos(**args)


def _main(job_folder, out_folder, video_args, labels, cores=1):

    arg_list = [
        {
            'mode': mode, 'filename': filename, 'labels': labels,
            'outdir': outdir, 'name': name, **video_args[mode]
        }
        for mode, name, filename, outdir in get_files(job_folder, out_folder)
        if mode in video_args
    ]

    if cores == 1:
        for al in arg_list:
            make_videos(al)
    else:
        pool = Pool(cores)
        pool.map(make_videos, arg_list)


def main():
    arguments = docopt(__doc__)
    run_folder = arguments['RUN_FOLDER']
    in_folder = arguments['IN_FOLDER']

    print(in_folder)

    parameter_file = os.path.join(run_folder, 'video.yml')

    out_folder = os.path.join(run_folder, 'video')
    parameter = load_yaml(parameter_file)

    if in_folder:
        _main(job_folder=in_folder, out_folder=out_folder, cores=parameter['exec']['cores'],
              labels=parameter.get('labels', {}), **parameter['params'])
    else:
        _main(job_folder=run_folder, out_folder=out_folder, cores=1,
              labels=parameter.get('labels', {}), **parameter['params'])


if __name__ == "__main__":
    main()
