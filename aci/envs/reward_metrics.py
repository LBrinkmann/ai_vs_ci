import torch as th
from aci.utils.tensor_op import map_tensor

METRIC_NAMES = [f'{agg}_{m}' for agg in ['ind', 'local', 'global']
                for m in ['try', 'crosscoordination', 'coordination', 'anticoordination']]


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
    _neighbors_mask = neighbors_mask.unsqueeze(
        1).expand(-1, s, -1, -1)  # h s+ p n+

    # note change in index (p -> n`), actions will be mapped on neighbors
    nb_tensor = tensor  # h s+ n` t ?
    nb_tensor = nb_tensor.unsqueeze(2)  # h s+ * n` t ?

    nb_tensor = nb_tensor.expand(-1, -1, p, *
                                 [-1]*(len(tensor.shape) - 2))  # h s+ p n` t ?

    _neighbors = _neighbors.clone()
    _neighbors[_neighbors_mask] = 0

    nb_tensor = map_tensor(_neighbors, nb_tensor)
    nb_tensor[_neighbors_mask] = fill_value  # h s+ p n+ t ?
    return nb_tensor  # h s+ p n+ t ?


def calc_metrics(actions, neighbors, neighbors_mask, null_action):
    """
    Calculates three different metrics (cross-coordination, coordination and
    anti-coordination). Each metric is calculated for each agent type individual
    and aggregated in three different ways (no-aggregation, local aggregation
    and global aggregation). Metrics are broadcasted in the case of global aggregation.

    Metrics:
        cross-coordination: 1 if color of both agents on the same node matches
        coordination: 1 if color of agent matches all neighbors of same type
        anti-coordination: 1 if color of agent mismatches all neighbors of same type

    Aggregations:
        individual: no aggregation, each node has its own unique value
        local: aggregation for each node accross itself and its neighbors
        global: aggregation over all nodes (of one agent type)

    Args:
        actions: h s+ p t
        neighbors: h p n+
        neighbors_mask: h p n+
    Returns:
    """
    h, s, p, t = actions.shape

    neighbors_actions = project_on_neighbors(
        actions, neighbors, neighbors_mask, fill_value=-1)  # h s+ p n t

    self_cross = actions  # h s+ p t
    other_cross = th.flip(actions, [3])  # h s+ p t
    cross_coordination = (self_cross == other_cross).type(th.float)
    if null_action:
        # cross_coordination = (2 * cross_coordination - 1.) * ((
        #     self_cross != 0) & (other_cross != 0)).type(th.float)   # h s+ p t
        cross_coordination = (2 * cross_coordination - 1.) * (
            self_cross != 0).type(th.float)   # h s+ p t

    _mask = neighbors_mask.unsqueeze(
        1).unsqueeze(-1).expand(-1, s, -1, -1, t)  # h s+ p n t

    if null_action:
        neighbor_not_null = neighbors_actions != 0  # h s+ p n t
        neighbor_not_null[_mask] = True  # h s+ p n t
        neighbor_not_null = neighbor_not_null.all(-2)

    neighbor_coordination = (actions.unsqueeze(-2)
                             == neighbors_actions)  # h s+ p n t
    neighbor_coordination[_mask] = True  # h s+ p n t
    # h s+ p t
    neighbor_coordination = neighbor_coordination.all(-2).type(th.float)

    if null_action:
        neighbor_coordination = (
            2 * neighbor_coordination - 1) * ((actions != 0) & neighbor_not_null).type(th.float)  # h s+ p  t

    only_neigbhors = neighbors_actions[:, :, :, 1:]
    neighbor_anticoordination = (actions.unsqueeze(-2) !=
                                 only_neigbhors)  # h s+ p n t
    # h s+ p t
    neighbor_anticoordination = neighbor_anticoordination.all(
        -2).type(th.float)
    if null_action:
        neighbor_anticoordination = (
            2 * neighbor_anticoordination - 1) * ((actions != 0) & neighbor_not_null).type(th.float)  # h s+ p  t

    is_not_null = (actions != 0).type(th.float)
    ind_metrics = th.stack([is_not_null, cross_coordination, neighbor_coordination,
                            neighbor_anticoordination], dim=-1)  # h s+ p t m

    loc_metrics = local_aggregation(ind_metrics, neighbors, neighbors_mask)
    glob_metrics = global_aggregation(ind_metrics)

    metrics = th.cat([ind_metrics, loc_metrics, glob_metrics], dim=-1)
    return metrics


def global_aggregation(metrics):
    """
    Calculates a global aggregation of all nodes.

    Args:
        metrics: h s+ p t m
    Returns:
        global_metrics: h s+ p t m
    """
    h, s, p, t, m = metrics.shape
    return metrics.mean(2, keepdim=True).expand(-1, -1, p, -1, -1)


def local_aggregation(metrics, neighbors, neighbors_mask):
    """
    Calculates a local aggregation of a node and its neighbors.

    Args:
        metrics: h s+ p t m
        neighbors: h p n+
        neighbors_mask: h p n+
    Returns:
        local_metrics: h s+ p t m
    """
    metrics_norm = metrics / \
        (~neighbors_mask).sum(-1).unsqueeze(1).unsqueeze(3).unsqueeze(4)  # h s + p t m
    nb_metrics = project_on_neighbors(
        metrics_norm, neighbors, neighbors_mask)  # h s+ p n t m
    local_metrics = nb_metrics.sum(3)  # h s+ p t m
    return local_metrics  # h s+ p t m
