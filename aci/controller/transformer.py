import torch as th


def onehot(x, m, input_size, device):
    onehot = th.zeros(*x.shape, input_size, dtype=th.float32, device=device)
    x_ = x.clone()
    x_[m] = 0
    onehot.scatter_(-1, x_.unsqueeze(-1), 1)
    onehot[m] = 0
    return onehot
