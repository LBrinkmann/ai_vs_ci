

def shift_obs(tensor_dict):
    """
    Creates previous and current observations.

    Args:
        tensor_dict: each tensor need to have the episode dimension at second position
    """
    previous = {k: v[:, :-1] if v is not None else None for k, v in tensor_dict.items()}
    current = {k: v[:, 1:] if v is not None else None for k, v in tensor_dict.items()}
    return previous, current
