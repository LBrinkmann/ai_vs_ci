import torch as th
from aci.envs.network_game import NetworkGame
from .utils import random_step
import warnings


def _test_obs_shape(config, agent_type):
    device = th.device('cpu')
    env = NetworkGame(**config, device=device)
    env.init_episode()
    rewards, done, info = random_step(env)
    obs_shapes = env.get_observation_shapes(agent_type=agent_type)

    # TODO: matrix seems to need more work
    obs_shapes.pop('matrix')
    warnings.warn("Matrix observations will not be tested.")

    for mode, shapes in obs_shapes.items():
        obs = env.observe(agent_type=agent_type, mode=mode)

        if not isinstance(obs, tuple):
            obs = (obs,)
            shapes = (shapes,)
        assert len(obs) == len(shapes), f'test_obs_shape, {mode} {agent_type}'
        for shape, o in zip(shapes, obs):
            if isinstance(shape, dict):
                assert o.shape == shape['shape']
                assert o.max() <= shape['maxval']
            elif shape is None:
                assert o is None
            else:
                assert o.shape == shape
