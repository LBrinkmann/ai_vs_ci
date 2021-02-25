import torch as th

from aci.controller.selector import eps_greedy


def test_eps_greedy():
    size = 10000
    eps = 0.01
    expected_range = [0.005, 0.015]

    ref_int = th.randint(0, 2, (size,))
    q = th.rand((size, 3))
    q[th.arange(size), ref_int] = 2

    q = q.unsqueeze(0).expand(6, -1, -1)

    action = eps_greedy(q, eps)

    ref_int = ref_int.unsqueeze(0).expand(6, -1)
    match = (action == ref_int).float().mean()
    miss = (1 - match)

    assert miss > expected_range[0]
    assert miss < expected_range[1]
