import torch as th


def select_from_history(state, neighbors, neighbors_mask, ci_ai_map, episode_idx, **other):
    return {
        # **other,
        'state': state[:, :, episode_idx],
        'neighbors': neighbors[:, episode_idx],
        'neighbors_mask': neighbors_mask[:, episode_idx],
        'ci_ai_map': ci_ai_map[:, episode_idx]
    }


def add_dimensions_to_current(state, neighbors, neighbors_mask, ci_ai_map, **other):
    return {
        # **other,
        'state': state.unsqueeze(-1).unsqueeze(-1),
        'neighbors': neighbors.unsqueeze(1),
        'neighbors_mask': neighbors_mask.unsqueeze(1),
        'ci_ai_map': ci_ai_map.unsqueeze(-1)
    }


def neighbors_states(state, neighbors, neighbors_mask, n_nodes, **_):
    episode_steps = state.shape[-1]

    state = state.permute(2, 3, 1, 0).unsqueeze(0).repeat(n_nodes, 1, 1, 1, 1)

    _neighbors = neighbors.clone()
    _neighbors[neighbors_mask] = 0
    _neighbors = _neighbors.unsqueeze(
        2).unsqueeze(-1).repeat(1, 1, episode_steps, 1, 2)
    neighbors_mask = neighbors_mask.unsqueeze(-2).repeat(1, 1, episode_steps, 1)

    neighbors_states = th.gather(state, -2, _neighbors)
    neighbors_states[neighbors_mask] = -1

    return neighbors_states, neighbors_mask


def shift_obs(*tenors, **_):
    """
    """
    previous = (t[:, :, :-1] for t in tenors)
    current = (t[:, :, 1:] for t in tenors)
    return previous, current


def map_ci_agents(neighbors_states, neighbor_mask, *, ci_ai_map, agent_type, **_):
    if agent_type == 'ci':
        pass
    elif agent_type == 'ai':
        _ci_ai_map = ci_ai_map \
            .unsqueeze(-1) \
            .unsqueeze(-1) \
            .unsqueeze(-1) \
            .repeat(1, 1, *neighbors_states.shape[2:])
        neighbors_states = th.gather(neighbors_states, 0, _ci_ai_map)
        neighbor_mask = th.gather(neighbor_mask, 0, _ci_ai_map[:, :, :, :, 0])
    else:
        raise ValueError(f'Unkown agent_type {agent_type}')
    return neighbors_states, neighbor_mask
