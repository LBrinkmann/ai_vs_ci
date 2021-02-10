import torch.nn as nn
import torch.nn.functional as F
import torch as th


class PoolingGRUAgent(nn.Module):
    """
    A agent based on two sandwich of linear and recurrent layer.

    Args:
        n_neighbors: Number of (potential) neighbors. This includes the agent itself.
        view_size: Size of the view encoding.
        control_size: Size of the control encoding.
        hidden_size: Size of the hidden encoding.
        n_actions: Number of actions.
        merge_pos: Position in the layer stack at which to merge the control encoding.
        merge_type: Strategy used to merge the control encoding.

    Indices:
        a: actions [0..n_actions]
        s: episode step [0..episode_steps (+ 1)]
        b: batch idx [0..batch_size]
        n: neighbors [0..max_neighbors (+ 1)]
        i: view encoding [0..n_view]
        c: control encoding [0..n_control]
        h: hidden encoding
    """

    def __init__(
            self, n_neighbors, view_size, control_size, n_actions, hidden_size, pooling_types,
            linear1, rnn1, linear2, rnn2, device, merge_pos=None, merge_type=None):
        super(PoolingGRUAgent, self).__init__()
        self.pooling_types = pooling_types
        self.n_actions = n_actions
        self.merge_pos = merge_pos
        self.merge_type = merge_type

        n_hidden = view_size
        if linear1:
            self.linear1 = nn.Linear(in_features=n_hidden, out_features=hidden_size)
            n_hidden = hidden_size
        else:
            self.linear1 = None

        if rnn1:
            self.rnn1 = nn.GRU(input_size=n_hidden, hidden_size=hidden_size, batch_first=True)
            n_hidden = hidden_size
        else:
            self.rnn1 = None

        # pooling
        n_hidden = n_hidden*(len(pooling_types) + 1)

        if merge_pos == 'prelin2':
            n_hidden = merged_size(n_hidden, control_size, merge_type)
        if linear2:
            self.linear2 = nn.Linear(in_features=n_hidden, out_features=hidden_size)
            n_hidden = hidden_size
        else:
            self.linear2 = None
        if merge_pos == 'prernn2':
            n_hidden = merged_size(n_hidden, control_size, merge_type)
        if rnn2:
            self.rnn2 = nn.GRU(
                input_size=n_hidden, batch_first=True,
                hidden_size=hidden_size)
            n_hidden = hidden_size
        else:
            self.rnn2 = None
        if merge_pos == 'prelin3':
            n_hidden = merged_size(n_hidden, control_size, merge_type)
        self.linear3 = nn.Linear(in_features=n_hidden, out_features=n_actions)

    def reset(self):
        self.hidden1 = None
        self.hidden2 = None

    def forward(self, view, mask, control=None):
        """
        Args:
            view: The view of the agent on itself and other agents. A 4-D `Tensor` of type
            `th.float32` and shape `[b,s,n,i]`.
            mask: A mask indicating valid values in the input. A 3-D `Tensor` of type
            `th.bool` and shape `[b,s,n]`.
            control: Additional input to control the agent. A 3-3 `Tensor` of type
            `th.float32` and shape `[b,s,c]`
        Indices:
            a: actions [0..n_actions]
            s: episode step [0..episode_steps (+ 1)]
            b: batch idx [0..batch_size]
            n: neighbors [0..max_neighbors (+ 1)]
            i: view encoding [0..n_view]
            c: control encoding [0..n_control]
            h: hidden encoding
        """
        b, s, n, _ = view.shape
        b2, s2, n2 = mask.shape

        assert b == b2
        assert s == s2
        assert n == n2

        view = view.permute(0, 2, 1, 3)  # b, n+, s, i
        mask = ~mask.permute(0, 2, 1).unsqueeze(-1)  # b, n+, s, *

        if self.linear1:
            view = F.relu(self.linear1(view))  # b*n+, s, h

        if self.rnn1:
            view = view.reshape(b*n, s, -1)  # b*n+, s, h
            view, self.hidden1 = self.rnn1(view, self.hidden1)  # b*n+, s, h
            view = view.reshape(b, n, s, -1)  # b, n+, s, h

        view = pool(view, mask, self.pooling_types)  # b, s, h

        if self.merge_pos == 'prelin2':
            view = merge_control(view, control, self.merge_type)  # b, s, h
        if self.linear2:
            view = F.relu(self.linear2(view))
        if self.merge_pos == 'prernn2':
            view = merge_control(view, control, self.merge_type)  # b, s, h
        if self.rnn2:
            view, self.hidden2 = self.rnn2(view, self.hidden2)  # b, s, h
        if self.merge_pos == 'prelin3':
            view = merge_control(view, control, self.merge_type)  # b, s, h
        q = self.linear3(view)  # b, s, a
        return q  # b, s, a


def pool(view, mask, pooling_types):
    """
    Args:
        view: `[b, n+, s, i]`
        mask: `[b, n+, s]`
    """
    # the first element in the neighbors dimension is representing the agent itself
    view_self = view[:, 0]  # b, s, h
    # the other elements are representing neighbors
    view_others = view[:, 1:] * mask[:, 1:]  # b, n, s, h

    pooled = [view_self]

    if 'avg' in pooling_types:
        pooled.append(view_others.sum(dim=1) / mask.sum(dim=1))  # b, s, h
    if 'max' in pooling_types:
        pooled.append(view_others.max(dim=1)[0])  # b, s, h
    if 'sum' in pooling_types:
        pooled.append(view_others.mean(dim=1))  # b, s, h

    return th.cat(pooled, axis=-1)  # b, s, i


def merged_size(hidden_size, control_size, merge_type):
    """
    Calculates the hidden size after the merge operation.
    """
    if merge_type == 'cat':
        return hidden_size + control_size
    elif merge_type == 'outer':
        return hidden_size * control_size
    elif merge_type == 'outer_normalized':
        return hidden_size * control_size
    else:
        raise NotImplementedError(f'Unkown merge type: {merge_type}')


def merge_control(view, control, merge_type):
    """
    Args
        view: `[b,s,h]`
        control: `[b,s,c]`
    """
    if merge_type == 'cat':
        return th.cat([view, control], axis=-1)  # b,s,h+c
    elif merge_type == 'outer':
        return th.einsum('ijk,ijl->ijkl', view, control) \
            .reshape(view.shape[0], view.shape[1], -1)  # b,s,h*c
    elif merge_type == 'outer_normalized':
        control_normed = control / control.sum(dim=-1, keepdim=True)  # b,s,c
        return th.einsum('ijk,ijl->ijkl', view,
                         control_normed).reshape(view.shape[0], view.shape[1], -1)  # b,s,h*c
    else:
        raise NotImplementedError(f'Unkown merge type: {merge_type}')
