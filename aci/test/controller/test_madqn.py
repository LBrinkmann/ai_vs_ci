import torch as th
import numpy as np
from pytest import approx

from aci.controller.madqn import GRUAgentWrapper


obs_schapes = {'shape': (3, 2, 2), 'maxval': 4}, {'shape': (3, 2), 'maxval': 1}, None, {
    'shape': (3, 2), 'maxval': 1, 'names': ['m1', 'm2']}

FIXED = {
    "observation_shapes": obs_schapes,
    "n_agents": 3,
    "n_actions": 4,
    "net_type": 'pooling_gru',
    "device": th.device('cpu'),
}

MODEL = {
    'linear1': True, 'rnn1': True, 'linear2': True, 'rnn2': True,
    'hidden_size': 20, 'pooling_types': ['avg', 'max', 'sum']
}

DEFAULT = {
    "multi_type": 'shared_weights',
    "add_catch": False,
    "control_type": None,
    "global_input": [],
    "global_control": [],
    "mix_weights_args": None,
    "action_permutation": False,
}


def test_multi_type():
    params = {**DEFAULT, **FIXED, **MODEL, "multi_type": 'shared_weights'}
    wrapper = GRUAgentWrapper(**params)
    print(wrapper.control_size)
    # x = th.Tensor(
    #     [
    #         [[[[0,0],[1,0]]],
    #         [[[[0,0],[1,0]]],
    #         [[[[0,0],[1,0]]],
    #     ]
    # )
    # mask = th.Tensor(
    #     [
    #         [[[[1,1],[1,1]]],
    #         [[[[1,1],[1,1]]],
    #         [[[[1,1],[1,1]]],
    #     ]
    # )
    # secret = th.Tensor(
    #     [
    #         [[1]],
    #         [[0]],
    #         [[1]],
    #     ]
    # )
    # globalx = th.Tensor(
    #     [
    #         [[[0.2,0.5]]],
    #         [[[0.2,0.5]]],
    #         [[[0.2,0.5]]],
    #     ]
    # )
    # wrapper()
