import torch as th


def map_tensor(map_idx, tensor):
    """
    Args:
        map_idx: Tensor type `th.long`. The last dim is going to be mapped.
        tensor: Tensor to be mapped. Shape must correspond with map_idx, in all dims but the last.
    Returns:
        mapped_tensor: Shape is the one of map_idx plus the remaining dims of tensor
    """
    m_dim = len(map_idx.shape)
    assert map_idx.shape[:-1] == tensor.shape[:m_dim-1]
    for s in tensor.shape[m_dim:]:
        map_idx = map_idx.unsqueeze(-1)
    map_idx = map_idx.expand(*[-1]*m_dim, *tensor.shape[m_dim:])
    return th.gather(tensor, m_dim-1, map_idx)


def map_tensors(map_idx, *tensors):
    return (map_tensor(map_idx, t) if t is not None else None for t in tensors)


def map_tensors_back(map_idx, *tensors):
    inv_map_idx = th.argsort(map_idx, dim=-1)
    return (map_tensor(inv_map_idx, t) if t is not None else None for t in tensors)
