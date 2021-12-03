import torch as th
from aci.observer.encoder import ViewEncoder


class NetworkView():
    def __init__(self, *, agent_type, view_args={}, env_info, device):
        self.device = device
        self.agent_type = agent_type
        self.agent_type_idx = env_info['agent_types'].index(agent_type)

        self.node_encoder = ViewEncoder(**view_args, env_info=env_info, device=device)

        n_nodes = env_info['n_nodes']
        max_neighbors = env_info['max_neighbors']

        self.shape = {
            'view': (-1, -1, n_nodes, max_neighbors, self.node_encoder.shape),
            'edge_index': (2, max_neighbors*n_nodes),
        }

    def __call__(self, actions, neighbors, neighbors_mask, **state):
        h, s, p, t = actions.shape

        view = self.node_encoder(**{'actions': actions, 'neighbors': neighbors,
                                    'neighbors_mask': neighbors_mask, **state})['view']  # h s+ p i

        h, p, n = neighbors.shape

        assert th.equal(
            neighbors_mask[0, :, 1:], neighbors_mask[-1, :, 1:]
        ), 'mask must be identical in batch.'
        target = neighbors[0, :, 1:]
        source = neighbors[0, :, [0]].expand(-1, n-1)
        mask = neighbors_mask[0, :, 1:].reshape(-1)

        edge_index = th.stack([source.reshape(-1), target.reshape(-1)])
        edge_index = edge_index[:, mask == 0]

        return {
            'view': view,   # h s+ p n i
            'edge_index': edge_index,  # h 2 e
        }
