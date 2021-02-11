import torch as th

METRIC_NAMES = [f'{agg}_{m}' for agg in ['ind', 'local', 'avg']
                for m in ['crosscoordination', 'coordination', 'anticoordination']]


def calc_metrics_reward(state, neighbors, neighbors_mask, n_nodes, reward_vec):
    """
    Args:
        state: h s+ p t
        neighbors: h p n+
        neighbors_mask: h p n+
        reward_vec: m t
    Returns:
        metrics: h s+ p m t
        rewards: h s+ p t
    """
    metrics = calc_metrics(neighbors_states, neighbors, neighbors_mask)  # h s+ p m t
    reward = calc_reward(metrics, reward_vec)  # h s+ p t
    return metrics, reward


def create_reward_vec(agent_types, device, **at_metrics):
    """
    Returns:
        reward_vec: A 2-D matrix of type `th.float` and shape `[t,m,t']` to map
            from the metrics to the reward. `t` is the agent type of the reward,
            `t'` is the agent type of the metric.
    """
    return th.tensor([
        [at_metrics[at][m] for m in METRIC_NAMES]
        for at in agent_types
    ], dtype=th.float, device=device)


def calc_reward(metrics, reward_vec):
    """
    Args:
        metrics: h s+ p m t
        reward_vec: m t
    Returns:
        reward: h s+ p t
    """
    reward = th.einsum('hspmt,mt->hspt', metrics, reward_vec)  # h s+ p t
    return reward


def project_on_neighbors(tensor, neighbors, neighbors_mask):
    """
    Args:
        tensor: h s+ p t ?
        neighbors: h p n+
        neighbors_mask: h p n+
    Returns:
        nb_tensor: h s+ p n t
    """
    h, s, p, t = tensor.shape[:4]
    _neighbors = neighbors.unsqueeze(1).unsqueeze(-1).expand(-1, s, -1, -1, t)  # h s+ p n+ t
    _neighbors_mask = neighbors_mask.unsqueeze(1).expand(-1, s, -1, -1)  # h s+ p n+

    # note change in index (p -> n`), states will be mapped on neighbors
    nb_tensor = tensor  # h s+ n` t ?
    nb_tensor = nb_tensor.unsqueeze(2)  # h s+ * n` t ?
    nb_tensor = nb_tensor.expand(-1, -1, p, -1, -1)  # h s+ p n` t ?

    _neighbors = neighbors.clone()
    _neighbors[_neighbors_mask] = 0

    nb_tensor = th.gather(nb_tensor, 3, _neighbors)  # h s+ p n+ t ?
    nb_tensor[_neighbors_mask] = -1  # h s+ p n+ t ?

    return nb_tensor  # h s+ p n+ t ?


def calc_metrics(state, neighbors, neighbors_mask):
    """
    Args:
        state: h s+ p t
        neighbors: h p n+
        neighbors_mask: h p n+
    Returns:
    """
    h, s, p, t = state

    neighbors_states = project_on_neighbors(
        state, neighbors, neighbors_mask)  # h s+ p n t

    self_cross = state.unsqueeze(-1).expand(-1, -1, -1, -1, t)  # h s+ p t t'
    other_cross = state.unsqueeze(-2).expand(-1, -1, -1, t, -1)  # h s+ p t t'
    cross_coordination = (self_cross == other_cross).all(-1).type(th.float)  # h s+ p t

    neighbor_coordination = (state.unsqueeze(-2) ==
                             neighbors_states).all(-2).type(th.float)  # h s+ p t
    neighbor_anticoordination = (state.unsqueeze(-2) !=
                                 neighbors_states).all(-2).type(th.float)  # h s+ p t
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
    local_metrics = nb_metrics.sum(3) / (~neighbors_mask).sum(-1)  # h s+ p t m
    return local_metrics  # h s+ p t m
