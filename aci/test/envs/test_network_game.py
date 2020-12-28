from aci.envs.network_game import NetworkGame
from aci.test.utils import load_configs
from subtests.general import _test_general
from subtests.obs_shape import _test_obs_shape
from subtests.neighbors import _test_neighbors_v1, _test_neighbors_v2
from subtests.sample import _test_sampling_v1, _test_sampling_v2


def test_general():
    for config in load_configs(['basic'], __file__):
        _test_general(config)


def test_env_shape():
    for config in load_configs(['basic'], __file__):
        _test_obs_shape(config, 'ai')
        _test_obs_shape(config, 'ci')


def test_neighbors():
    for config in load_configs(['basic'], __file__):
        _test_neighbors_v1(config, 'ai')
        _test_neighbors_v1(config, 'ci')
        _test_neighbors_v2(config, 'ai')
        _test_neighbors_v2(config, 'ci')


def test_sample():
    for config in load_configs(['basic'], __file__):
        _test_sampling_v1(config, 'ai')
        _test_sampling_v1(config, 'ci')
        _test_sampling_v2(config, 'ai')
        _test_sampling_v2(config, 'ci')
