import torch as th

from aci.utils.tensor_op import map_tensors, map_tensors_back


def test_map_tensor():
    test = [
        [[0, 1, 2],
         [3, 4, 5],
         [10, 11, 12]],
        [[13, 14, 15],
         [20, 21, 22],
         [23, 24, 25]],
    ]
    _map = [
        [1, 0, 2],
        [2, 1, 0],
    ]
    ref = [
        [[3, 4, 5],
         [0, 1, 2],
         [10, 11, 12]],
        [[23, 24, 25],
         [20, 21, 22],
         [13, 14, 15]],
    ]
    test = th.tensor(test)
    _map = th.tensor(_map)
    ref = th.tensor(ref)
    mapped, = map_tensors(_map, test)
    mapped_back, = map_tensors_back(_map, mapped)

    assert (mapped == ref).all()
    assert (mapped_back == test).all()
