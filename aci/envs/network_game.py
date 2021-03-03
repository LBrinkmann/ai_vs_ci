"""
Graph Coloring Environment
"""

import torch as th
import numpy as np
import string
import os
from .utils.graph import create_graph, determine_max_degree
from aci.utils.memory import Memory
from ..utils.io import ensure_dir
from aci.envs.reward_metrics import calc_metrics, calc_reward, create_reward_vec, METRIC_NAMES


def create_map(n_nodes, device, mapping_type=None,  **_):
    if mapping_type == 'random':
        return th.randperm(n_nodes, device=device)
    elif mapping_type == 'fixed':
        return th.arange(n_nodes, device=device)
    else:
        raise ValueError('Unkown mapping type.')


def create_maps(agent_type_args, n_nodes, device):
    """
    Mapping between agent_idx and network position.

    Returns:
        mapping: A tensor of shape ``
    """
    return th.stack([
        create_map(**ata, n_nodes=n_nodes, device=device)
        for ata in agent_type_args.values()
    ], dim=-1)


def create_control(
        n_nodes, n_agent_types, n_control=None, correlated=None, cross_correlated=None,
        device=None, **kwargs):
    """
    Args:
        control: p, t
    """
    if n_control is None:
        return None
    assert (n_nodes is not None)
    assert (correlated is not None)
    assert (cross_correlated is not None)
    if correlated and cross_correlated:
        c = th.randint(low=0, high=n_control, size=(1, 1), device=device)\
            .expand(n_nodes, n_agent_types)
    if correlated and not cross_correlated:
        c = th.randint(low=0, high=n_control, size=(1, n_agent_types),
                       device=device).expand(n_nodes, -1)
    if not correlated and cross_correlated:
        c = th.randint(low=0, high=n_control, size=(n_nodes, 1),
                       device=device).expand(-1, n_agent_types)
    if not correlated and not cross_correlated:
        c = th.randint(low=0, high=n_control, size=(n_nodes, n_agent_types), device=device)
    return c.unsqueeze(0).unsqueeze(0)


def pad_neighbors(neighbors, max_degree, device):
    padded_neighbors = [
        [i] + n + [-1]*(max_degree - len(n))
        for i, n in enumerate(neighbors)
    ]
    mask = [
        [False]*(len(n)+1) + [True]*(max_degree - len(n))
        for i, n in enumerate(neighbors)
    ]
    return (
        th.tensor(padded_neighbors, dtype=th.int64, device=device),
        th.tensor(mask, dtype=th.bool, device=device)
    )


class NetworkGame():
    """
    An multiagent environment. Agents are placed on a undirected network. On each
    node on of two types of agents are placed. On each timestep agents can select a
    discrete action. Reward is depending on whether actions of agents on the same
    node match and whether actions of neighboring actions match.

    The environment keeps track of historic agents and allows to resample from those.
    Periodically this history is saved, which allows to reconstruct the dynamic.


    Indices:
        t: agent types [0..1]
        p: positions / nodes [0..n_nodes]
        a: actions [0..n_actions]
        s: episode step [0..episode_steps (+ 1)]
        h: history position [0..max_history]
        b: batch idx [0..batch_size]
        n: neighbors [0..max_neighbors (+ 1)]
        m: metrics [0..len(metric_names)]
    """

    def __init__(
            self, *, n_nodes, n_actions, episode_steps, max_history,
            network_period=0, mapping_period=0,
            reward_args, agent_type_args, graph_args, control_args,
            out_dir=None, save_interval, device):
        """
        Args:
            n_nodes: Number of nodes of the network.
            n_actions: Number of actions available to each agent.
            episode_steps: Number of steps for one episode.
            max_history: Maximum number of episodes to memorize.
            network_period: Period of episodes to resample a new network.
            mapping_period: Period of episodes to remap agents between the two types.
            agent_type_args: Settings for each agent type.
            control_args: Settings for the control integer.
            graph_args: Settings for the network creation.
            out_dir: Folder to store data.
            device: A torch device.
        """
        self.device = device

        self.mapping_period = mapping_period
        self.network_period = network_period
        self.max_history = max_history
        self.graph_args = graph_args
        self.agent_types = list(agent_type_args.keys())
        self.agent_types_idx = {v: i for i, v in enumerate(self.agent_types)}
        self.agent_type_args = agent_type_args
        self.metric_names = METRIC_NAMES

        self.episode_steps = episode_steps
        self.n_nodes = n_nodes
        self.n_actions = n_actions
        self.n_agent_types = len(self.agent_types)
        self.control_args = control_args
        self.n_control = self.control_args.get('n_control', 0)
        # maximum number of neighbors
        self.max_neighbors = determine_max_degree(n_nodes=n_nodes, **graph_args)

        self.episode = -1
        self.out_dir = out_dir
        if out_dir is not None:
            ensure_dir(out_dir)

        self.episode_history = Memory(max_history, None, device)
        self.step_history = Memory(max_history, episode_steps+1, device)
        self.save_interval = save_interval
        self.episode_state = {}
        self.step_state = {}

        self.reward_vec = create_reward_vec(reward_args, self.device)

        self.info = {
            'n_agent_types': self.n_agent_types,
            'n_actions': self.n_actions,
            'n_nodes': self.n_nodes,
            'n_control': self.n_control,
            'max_neighbors': self.max_neighbors,
            'episode_steps': self.episode_steps,
            'metric_names': self.metric_names,
            'actions': [string.ascii_uppercase[i] for i in range(self.n_actions)],
            'agents': [string.ascii_uppercase[i] for i in range(self.n_nodes)],
            'agent_types': self.agent_types,
            'metric_names': self.metric_names,
        }

    def init_episode(self):
        self.episode += 1
        self.episode_step = -1
        self.episode_state['episode'] = th.tensor(
            self.episode, dtype=th.int64, device=self.device).unsqueeze(0)

        # create new graph (either only on first episode, or every network_period episode)
        if (self.episode == 0) or (
                (self.network_period > 0) and (self.episode % self.network_period == 0)):
            neighbors = create_graph(n_nodes=self.n_nodes, **self.graph_args)
            neighbors, neighbors_mask = pad_neighbors(neighbors, self.max_neighbors, self.device)
            self.episode_state['neighbors'] = neighbors.unsqueeze(0)
            self.episode_state['neighbors_mask'] = neighbors_mask.unsqueeze(0)

            # mapping between network and agents
        if (self.episode == 0) or (
                (self.mapping_period > 0) and (self.episode % self.mapping_period == 0)):
            self.episode_state['agent_map'] = create_maps(
                self.agent_type_args, self.n_nodes, self.device).unsqueeze(0)

        self.episode_history.start_episode(self.episode)
        self.step_history.start_episode(self.episode)
        # squeezed_episode_state = {
        #     k: v.squeeze(0) for k, v in self.step_state.items()
        # }
        self.episode_history.add(self.step_state)

        # random init action
        init_actions = {
            at: th.tensor(
                np.random.randint(0, self.n_actions, (1, 1, self.n_nodes)),
                dtype=th.int64, device=self.device
            )
            for at in self.agent_types
        }

        # init step
        return (init_actions, *self.step(init_actions)[:2])

    def step(self, actions):
        actions = th.stack([actions[at] for at in self.agent_types], dim=-1)

        self.episode_step += 1

        assert actions.max() <= self.n_actions - 1
        assert actions.dtype == th.int64

        if (self.episode_step == self.episode_steps):
            done = True
        elif self.episode_step > self.episode_steps:
            raise ValueError('Environment is done already.')
        else:
            done = False

        neighbors = self.episode_state['neighbors']
        neighbors_mask = self.episode_state['neighbors_mask']

        metrics = calc_metrics(actions, neighbors, neighbors_mask)  # h s+ p m t
        reward = calc_reward(metrics, self.reward_vec)
        control_int = create_control(
            n_nodes=self.n_nodes, n_agent_types=self.n_agent_types, **self.control_args,
            device=self.device)  # h s+ p t

        step_state = {
            'metrics': metrics,
            'reward': reward,
            'actions': actions,
            'control_int': control_int
        }
        # squeezed_step_state = {
        #     k: v.squeeze(0).squeeze(0) for k, v in step_state.items()
        # }

        self.step_history.add(step_state, self.episode_step)

        state = {**step_state, **self.episode_state}
        reward = {at: reward[:, :, :, at_idx] for at_idx, at in enumerate(self.agent_types)}
        return state, reward,  done

    def write_history(self):
        if self.out_dir is not None:
            filename = os.path.join(self.out_dir, f"{self.episode + 1}.pt")
            th.save(
                {
                    'agent_types': self.agent_types,
                    'actions': [string.ascii_uppercase[i] for i in range(self.n_actions)],
                    'agents': [string.ascii_uppercase[i] for i in range(self.n_nodes)],
                    'metric_names': self.metric_names,
                    **self.episode_history.stride(self.save_interval),
                    **self.step_history.stride(self.save_interval),
                },
                filename
            )

    def finish_episode(self):
        if self.episode_history.finished():
            assert self.step_history.finished()
            self.write_history()

    def __del__(self):
        if not self.episode_history.finished():
            assert not self.step_history.finished()
            self.write_history()
