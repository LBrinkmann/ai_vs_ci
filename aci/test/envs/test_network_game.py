import torch as th
import numpy as np
import os
from pytest import approx

from aci.envs.network_game import NetworkGame
from aci.test.utils import load_configs
from subtests.general import general_test


def test_env_basics():
    for config in load_configs(['basic'], __file__):
        general_test(config)
