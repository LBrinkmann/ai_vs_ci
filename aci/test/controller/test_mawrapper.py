import torch as th

from aci.controller.mawrapper import apply_permutations, create_permutations


def test_permutations():
    device = th.device('cpu')
    permutations = create_permutations(2, device)
    assert (permutations == th.tensor([[0, 1], [1, 0]])).all()
    control_int = [0, 1, 1]
    test = [
        [0, 1],
        [3, 4],
        [10, 11]
    ]
    ref = [
        [0, 1],
        [4, 3],
        [11, 10]
    ]
    control_int = th.tensor(control_int)
    test = th.tensor(test)
    ref = th.tensor(ref)
    permuted = apply_permutations(test, control_int, permutations)
    assert (permuted == ref).all()
