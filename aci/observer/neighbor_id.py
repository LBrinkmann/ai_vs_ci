import torch as th
import itertools
from aci.envs.reward_metrics import project_on_neighbors


def create_neighbor_combinations(max_neighbors, n_actions):
    actions = list(range(-1, n_actions))
    return list(itertools.product(
        *itertools.repeat(actions, max_neighbors)))


def create_self_combinations(n_actions):
    return list(range(n_actions))


def create_neighbor_map(max_neighbors, n_actions):
    neighbor_comb = create_neighbor_combinations(max_neighbors, n_actions)
    neighbor_comb_sorted = [sorted(n) for n in neighbor_comb]
    neighbor_comb_sorted = list(set(n for n in neighbor_comb_sorted))
    neighbor_dict = {t: i for i, t in enumerate(neighbor_comb_sorted)}

    return neighbor_dict


def map_neighbors(projected_actions, neighbor_dict):
    return th.tensor([
        [
            [
                neighbor_dict(p.tolist())
                for p in s]
            for s in e
        ]
        for e in projected_actions
    ])


def map_self(self_actions, self_dict):
    return th.tensor([
        [
            [
                self_dict(p.item())
                for p in s]
            for s in e
        ]
        for e in self_actions
    ])


class NeighborIDView():
    def __init__(self, *, agent_type, self_agent_type, neighbor_agent_type, env_info, device):
        self.device = device
        self.agent_type = agent_type

        n_nodes = env_info['n_nodes']
        n_actions = env_info['n_actions']
        max_neighbors = env_info['max_neighbors']
        self.self_agent_type_idx = env_info['agent_types'].index(self_agent_type)
        self.neighbor_agent_type_idx = env_info['agent_types'].index(neighbor_agent_type)

        self.self_map = {i: i for i in range(n_actions)}
        self.neighbor_map = create_neighbor_map(max_neighbors, n_actions)

        self.shape = {
            'self_id': (-1, -1, n_nodes),
            'neighbor_id': (-1, -1, n_nodes),
        }

    def __call__(self, **state):
        h, s, p, t = state['actions'].shape

        self_actions = state['actions'][:, :, :, self.self_agent_type_idx]
        projected_actions = project_on_neighbors(
            state['actions'][:, :, :, self.neighbor_agent_type_idx],
            state['neighbors'], state['neighbors_mask'])  # h s+ p n

        neighbors_id = map_neighbors(projected_actions, self.neighbor_dict)

        self_id = map_self(self_actions, self.self_dict)
        return {
            'neighbors_id': neighbors_id,   # h s+ p
            'self_id': self_id,  # h s+ p
        }
