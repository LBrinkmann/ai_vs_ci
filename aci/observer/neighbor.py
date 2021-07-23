from aci.envs.reward_metrics import project_on_neighbors
from aci.observer.encoder import ViewEncoder


class NeighborView():
    def __init__(self, *, agent_type, neighbor_view_args={}, control_view_args={}, env_info, device):
        self.device = device
        self.agent_type = agent_type
        self.agent_type_idx = env_info['agent_types'].index(agent_type)

        self.neighbor_encoder = ViewEncoder(**neighbor_view_args, env_info=env_info, device=device)
        self.control_encoder = ViewEncoder(**control_view_args, env_info=env_info, device=device)

        n_nodes = env_info['n_nodes']
        max_neighbors = env_info['max_neighbors']

        self.shape = {
            'view': (-1, -1, n_nodes, max_neighbors, self.neighbor_encoder.shape),
            'control': (-1, -1, n_nodes, self.control_encoder.shape),
            'mask': (-1, -1, n_nodes, max_neighbors),
            'agent_map': (-1, -1, n_nodes),
            'control_int': (-1, -1, n_nodes)
        }

    def __call__(self, **state):
        h, s, p, t = state['actions'].shape
        agent_type_idx = self.agent_type_idx

        view = self.neighbor_encoder(**state)['view']  # h s+ p i
        view = project_on_neighbors(
            view, state['neighbors'], state['neighbors_mask'])  # h s+ p n i
        control = self.control_encoder(**state)['view']  # h s+ p c

        mask = state['neighbors_mask'].unsqueeze(1).expand(-1, s, -1, -1)  # h s+ p n
        agent_map = state['agent_map'][:, :, agent_type_idx].unsqueeze(
            1).expand(-1, s, -1)  # h s+ p
        control_int = state['control_int'][:, :, :, agent_type_idx]  # h s+ p
        return {
            'view': view,   # h s+ p n i
            'control': control,  # h s+ p c
            'mask': mask,  # h s+ p n
            'agent_map': agent_map,  # h s+ p
            'control_int': control_int  # h s+ p
        }
