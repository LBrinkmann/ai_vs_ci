import torch as th


def select_from_history(state, neighbors, neighbors_mask, ci_ai_map, episode_idx, reward, metrics, secretes=None,  **other):
    return {
        # **other,
        'state': state[:, :, episode_idx],
        'neighbors': neighbors[:, episode_idx],
        'neighbors_mask': neighbors_mask[:, episode_idx],
        'ci_ai_map': ci_ai_map[:, episode_idx],
        'reward': reward[:, :, episode_idx],
        'secretes': secretes[:, episode_idx] if secretes is not None else None,
        'metrics': metrics[:, episode_idx]}


def add_dimensions_to_current(state, neighbors, neighbors_mask, ci_ai_map, **other):
    return {
        # **other,
        'state': state.unsqueeze(-1).unsqueeze(-1),
        'neighbors': neighbors.unsqueeze(1),
        'neighbors_mask': neighbors_mask.unsqueeze(1),
        'ci_ai_map': ci_ai_map.unsqueeze(-1)
    }


def neighbors_states(state, neighbors, neighbors_mask, n_nodes, **other):
    episode_steps = state.shape[-1]

    _state = state.permute(2, 3, 1, 0).unsqueeze(0).repeat(n_nodes, 1, 1, 1, 1)

    _neighbors = neighbors.clone()
    _neighbors[neighbors_mask] = 0
    _neighbors = _neighbors.unsqueeze(
        2).unsqueeze(-1).repeat(1, 1, episode_steps, 1, 2)
    neighbors_states_mask = neighbors_mask.unsqueeze(-2).repeat(1, 1, episode_steps, 1)

    neighbors_states = th.gather(_state, -2, _neighbors)
    neighbors_states[neighbors_states_mask] = -1

    return {
        **other,
        'neighbors': neighbors,
        'state': state,
        'neighbors_mask': neighbors_mask,
        'neighbors_states': neighbors_states,
        'neighbors_states_mask': neighbors_states_mask
    }


def shift_obs(tensor_dict, names, block):
    """
    """
    previous = (tensor_dict[n][:, :, :-1] if n not in block else None for n in names)
    current = (tensor_dict[n][:, :, 1:] if n not in block else None for n in names)
    return previous, current


# TODO: mapping of secrets, envstate, ... is missing
def map_ci_agents(neighbors_states, neighbors_states_mask, ci_ai_map, agent_type, state=None, reward=None,  **other):
    if agent_type == 'ci':
        pass
    elif agent_type == 'ai':
        _ci_ai_map = ci_ai_map \
            .unsqueeze(-1) \
            .unsqueeze(-1) \
            .unsqueeze(-1) \
            .repeat(1, 1, *neighbors_states.shape[2:])
        neighbors_states = th.gather(neighbors_states, 0, _ci_ai_map)
        neighbors_states_mask = th.gather(neighbors_states_mask, 0, _ci_ai_map[:, :, :, :, 0])

        # TODO: this is temporary
        if (state is not None) and (state.shape[2] > 1):

            _ci_ai_map = ci_ai_map.unsqueeze(0)\
                .unsqueeze(-1) \
                .expand(state.shape[0], -1, -1, state.shape[3])
            state = th.gather(state, 1, _ci_ai_map)
        else:
            state = None

        if reward is not None:
            _ci_ai_map = ci_ai_map.unsqueeze(0)\
                .unsqueeze(-1) \
                .expand(reward.shape[0], -1, -1, reward.shape[3])
            reward = th.gather(reward, 1, _ci_ai_map)
        else:
            reward = None
    else:
        raise ValueError(f'Unkown agent_type {agent_type}')
    return {
        **other,
        'neighbors_states': neighbors_states,
        'neighbors_states_mask': neighbors_states_mask,
        'state': state,
        'reward': reward
    }


def calc_metrics(neighbors_states, neighbors, neighbors_mask, **_):
    self_color = neighbors_states[:, :, :, 0]
    neighbor_color = neighbors_states[:, :, :, 1:]
    other_color = th.flip(self_color, [-1])

    catch = (self_color == other_color).type(th.float)
    # coordination = (self_color == neighbor_color).all(-1).type(th.float)
    anticoordination = (self_color.unsqueeze(-2) != neighbor_color).all(-2).type(th.float)
    ind_metrics = th.stack([catch, anticoordination], dim=-1)

    # TODO: temporary workaround
    loc_metrics = local_aggregation(ind_metrics, neighbors, neighbors_mask)
    glob_metrics = global_aggregation(ind_metrics)
    metrics = th.cat([ind_metrics, loc_metrics, glob_metrics], dim=-1)
    return metrics


#  (self.n_nodes, self.max_history, self.episode_steps + 1, len(self.metric_names)


def global_aggregation(metrics):
    n_nodes = metrics.shape[0]
    return metrics.mean(0, keepdim=True).expand(n_nodes, *[-1]*(len(metrics.shape)-1))


# TODO: this needs to be checked
def local_aggregation(metrics, neighbors, neighbors_mask):

    n_nodes, batch_size, episode_steps, n_agent_types, n_metrics = metrics.shape
    n_nodes2, batch_size_2, max_neighbors = neighbors.shape

    assert n_nodes == n_nodes2
    assert batch_size == batch_size_2

    # permutation needed because we map neighbors on agents
    _metric = metrics.unsqueeze(0).expand(n_nodes, -1, -1, -1, -1, -1).permute(1, 0, 2, 3, 4, 5)

    _neighbors = neighbors.clone()
    _neighbors[neighbors_mask] = 0
    _neighbors = _neighbors.permute(2, 0, 1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
        -1, -1, -1, episode_steps, n_agent_types, n_metrics)
    _neighbors_mask = neighbors_mask.permute(2, 0, 1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
        -1, -1, -1, episode_steps, n_agent_types, n_metrics)

    # agent_type, agents, batch, episode_step, neighbors
    local_metrics = th.gather(_metric, 0, _neighbors)
    local_metrics[_neighbors_mask] = 0
    local_metrics = local_metrics.sum(0) / (~_neighbors_mask).sum(0)
    return local_metrics
