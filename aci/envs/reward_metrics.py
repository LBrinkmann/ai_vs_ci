import torch as th
from aci.utils.tensor_op import map_tensor

METRIC_NAMES = [f'{agg}_{m}' for agg in ['ind', 'local', 'global']
                for m in ['crosscoordination', 'coordination', 'anticoordination']]

# print(METRIC_NAME)


def create_reward_vec(reward_args, device):
    """
    Returns:
        reward_vec: A 3-D matrix of type `th.float` and shape `[t,m,o']` to map
            from the metrics to the reward. `t` is the agent type of the reward,
            `t'` is the agent type of the metric.
    """
    assert all(m in METRIC_NAMES for ata in reward_args.values()
               for m in ata), "Unkown metric name in reward function."

    return th.tensor([
        [
            [ata.get(m, {}).get(o, 0) for o in reward_args.keys()]
            for m in METRIC_NAMES
        ]
        for ata in reward_args.values()
    ], dtype=th.float, device=device)


def calc_reward(metrics, reward_vec):
    """
    Args:
        metrics: h s+ p o m
        reward_vec: t m o
    Returns:
        reward: h s+ p t
    """
    reward = th.einsum('hspom,tmo->hspt', metrics, reward_vec)  # h s+ p t
    return reward


def project_on_neighbors(tensor, neighbors, neighbors_mask, fill_value=0):
    """
    Args:
        tensor: h s+ p t ?
        neighbors: h p n+
        neighbors_mask: h p n+
    Returns:
        nb_tensor: h s+ p n t ?
    """
    h, s, p, t = tensor.shape[:4]
    _neighbors = neighbors.unsqueeze(1).expand(-1, s, -1, -1)  # h s+ p n+
    _neighbors_mask = neighbors_mask.unsqueeze(1).expand(-1, s, -1, -1)  # h s+ p n+

    # note change in index (p -> n`), actions will be mapped on neighbors
    nb_tensor = tensor  # h s+ n` t ?
    nb_tensor = nb_tensor.unsqueeze(2)  # h s+ * n` t ?

    nb_tensor = nb_tensor.expand(-1, -1, p, *[-1]*(len(tensor.shape) - 2))  # h s+ p n` t ?

    _neighbors = _neighbors.clone()
    _neighbors[_neighbors_mask] = 0

    nb_tensor = map_tensor(_neighbors, nb_tensor)
    nb_tensor[_neighbors_mask] = fill_value  # h s+ p n+ t ?
    return nb_tensor  # h s+ p n+ t ?


def calc_metrics(actions, neighbors, neighbors_mask):
    """
    Args:
        actions: h s+ p t
        neighbors: h p n+
        neighbors_mask: h p n+
    Returns:
    """
    h, s, p, t = actions.shape

    neighbors_actions = project_on_neighbors(
        actions, neighbors, neighbors_mask, fill_value=-1)  # h s+ p n t

    self_cross = actions.unsqueeze(-1).expand(-1, -1, -1, -1, t)  # h s+ p t t'
    other_cross = actions.unsqueeze(-2).expand(-1, -1, -1, t, -1)  # h s+ p t t'
    cross_coordination = (self_cross == other_cross).all(-1).type(th.float)  # h s+ p t

    _mask = neighbors_mask.unsqueeze(1).unsqueeze(-1).expand(-1, s, -1, -1, t)  # h s+ p n t
    neighbor_coordination = (actions.unsqueeze(-2) == neighbors_actions)  # h s+ p n t
    neighbor_coordination[_mask] = True  # h s+ p n t
    neighbor_coordination = neighbor_coordination.all(-2).type(th.float)  # h s+ p t

    only_neigbhors = neighbors_actions[:, :, :, 1:]
    neighbor_anticoordination = (actions.unsqueeze(-2) !=
                                 only_neigbhors).all(-2).type(th.float)  # h s+ p t
    ind_metrics = th.stack([cross_coordination, neighbor_coordination,
                            neighbor_anticoordination], dim=-1)  # h s+ p t m

    loc_metrics = local_aggregation(ind_metrics, neighbors, neighbors_mask)
    glob_metrics = global_aggregation(ind_metrics)
    metrics = th.cat([ind_metrics, loc_metrics, glob_metrics], dim=-1)
    return metrics


def global_aggregation(metrics):
    """
    Args:
        metrics: h s+ p t m
    Returns:
        global_metrics: h s+ p t m
    """
    h, s, p, t, m = metrics.shape
    return metrics.mean(2, keepdim=True).expand(-1, -1, p, -1, -1)


def local_aggregation(metrics, neighbors, neighbors_mask):
    """
    Args:
        metrics: h s+ p t m
        neighbors: h p n+
        neighbors_mask: h p n+
    Returns:
        local_metrics: h s+ p t m
    """
    nb_metrics = project_on_neighbors(metrics, neighbors, neighbors_mask)  # h s+ p n t m
    local_metrics = nb_metrics.sum(
        3) / (~neighbors_mask.unsqueeze(1).unsqueeze(3).unsqueeze(4)).sum(-1)  # h s+ p t m
    return local_metrics  # h s+ p t m
