def shift_one(tensor, shape, prev):
    if tensor is None:
        return None
    elif shape[1] != -1:
        return tensor
    else:
        if prev:
            return tensor[:, :-1]
        else:
            return tensor[:, 1:]


def shift_obs(tensor_dict, obs_shape):
    """
    Creates previous and current observations.

    Args:
        tensor_dict: each tensor need to have the episode dimension at second position
    """
    previous = {k: shift_one(v, obs_shape[k], True) for k, v in tensor_dict.items()}
    current = {k: shift_one(v, obs_shape[k], False) for k, v in tensor_dict.items()}
    return previous, current
