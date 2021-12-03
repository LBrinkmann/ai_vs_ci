import torch as th


def eps_greedy(q_values, eps, device):
    """
    Args:
        q_values: Tensor of type `th.float` and arbitrary shape, last dimension reflect the actions.
        eps: fraction of actions sampled at random
    Returns:
        actions: Tensor of type `th.long` and the same dimensions then q_values, besides of the last.
    """
    n_actions = q_values.shape[-1]
    actions_shape = q_values.shape[:-1]

    greedy_actions = q_values.argmax(-1)
    random_actions = th.randint(0, n_actions, size=actions_shape, device=device)

    # random number which determine whether to take the random action
    random_numbers = th.rand(size=actions_shape, device=device)
    select_random = (random_numbers < eps).long()
    picked_actions = select_random * random_actions + (1 - select_random) * greedy_actions

    return picked_actions
