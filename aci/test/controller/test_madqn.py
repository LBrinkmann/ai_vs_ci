import torch as th
from pytest import approx

from aci.controller.madqn import shift_obs


def test_shift_obs():
    current = th.arange(1, 10).unsqueeze(0).unsqueeze(-1).expand(5, -1, 3)
    previous = th.arange(0, 9).unsqueeze(0).unsqueeze(-1).expand(5, -1, 3)
    test = th.arange(0, 10).unsqueeze(0).unsqueeze(-1).expand(5, -1, 3)
    t_previous, t_current = shift_obs({'test': test})

    assert (current == t_current['test']).all()
    assert (previous == t_previous['test']).all()
