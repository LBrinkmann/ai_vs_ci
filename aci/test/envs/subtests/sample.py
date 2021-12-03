import torch as th
from aci.envs.network_game import NetworkGame
from .utils import random_step
from itertools import count
import warnings


def _test_sampling_v1(config, agent_type):
    device = th.device('cpu')
    env = NetworkGame(**config, device=device)
    env.init_episode()

    n_nodes = env.n_nodes
    n_actions = env.n_actions

    BATCH_SIZE = 3

    test_observations = th.empty(
        (env.n_nodes, BATCH_SIZE, env.episode_steps, env.max_neighbors + 1, 2), dtype=th.int64)
    test_mask = th.empty((env.n_nodes, BATCH_SIZE, env.episode_steps,
                          env.max_neighbors + 1), dtype=th.int64)
    test_actions = th.empty((env.n_nodes, BATCH_SIZE, env.episode_steps), dtype=th.int64)
    test_rewards = th.empty((env.n_nodes, BATCH_SIZE, env.episode_steps), dtype=th.float32)

    for i in range(BATCH_SIZE):
        env.init_episode()
        for j in count():
            ai_color = th.randint(n_actions, size=(n_nodes,))
            ci_color = th.randint(n_actions, size=(n_nodes,))
            actions = {'ai': ai_color, 'ci': ci_color}
            rewards, done, info = env.step(actions)
            obs, mask = env.observe(agent_type=agent_type, mode='neighbors_with_mask')
            test_observations[:, 2-i, j] = obs
            test_mask[:, 2-i, j] = mask
            test_actions[:, 2-i, j] = actions[agent_type]
            test_rewards[:, 2-i, j] = rewards[agent_type]

            if done:
                break
    sample = env.sample(
        agent_type=agent_type,
        mode='neighbors_with_mask',
        batch_size=BATCH_SIZE,
        last=True
    )
    (prev_observations, prev_mask), (observations, mask), actions, rewards = sample

    assert prev_observations.shape[:2] == (env.n_nodes, BATCH_SIZE)
    assert (
        rewards.shape[:3] == actions.shape[:3] ==
        prev_observations.shape[:3] == observations.shape[:3] ==
        prev_mask.shape[:3] == mask.shape[:3]
    )
    assert (prev_observations[:, :, 1:] == observations[:, :, :-1]).all()
    assert (prev_mask[:, :, 1:] == mask[:, :, :-1]).all()

    assert (observations[:, :, :, 0, env.agent_types_idx[agent_type]] == actions).all()

    assert (test_observations == observations).all()
    assert (test_rewards == rewards).all()
    assert (test_actions == actions).all()
    assert (test_mask == mask).all()


def _test_sampling_v2(config, agent_type):
    device = th.device('cpu')
    BATCH_SIZE = 3
    N_SEEDS = 8

    # agent type gets secret; uncorrelated
    secrete_args = {
        'n_seeds': N_SEEDS,
        'correlated': False,
        'agent_types': [agent_type],
        'secret_period': 1
    }
    env = NetworkGame(**config, secrete_args=secrete_args, device=device)
    env.init_episode()
    obs_shapes = env.get_observation_shapes(agent_type=agent_type)['neighbors_mask_secret_envinfo']
    has_envinfo = obs_shapes[3] is not None
    if has_envinfo:
        envinfo_shape = obs_shapes[3]['shape'][1]
        test_envinfo = th.zeros(
            (env.n_nodes, BATCH_SIZE, env.episode_steps, envinfo_shape), dtype=th.float)

    test_secrets = th.zeros((env.n_nodes, BATCH_SIZE, env.episode_steps), dtype=th.int64)

    assert test_secrets.max() == 0

    for i in range(BATCH_SIZE):
        env.init_episode()
        for j in count():
            rewards, done, info = random_step(env)
            obs, mask, secrets, envinfo = env.observe(
                agent_type=agent_type, mode='neighbors_mask_secret_envinfo')

            if secrets is not None:
                test_secrets[:, BATCH_SIZE-i-1, j] = secrets

            if has_envinfo:
                test_envinfo[:, BATCH_SIZE-i-1, j] = envinfo
            else:
                assert envinfo is None

            if done:
                break

    sample = env.sample(
        agent_type=agent_type,
        mode='neighbors_mask_secret_envinfo',
        batch_size=BATCH_SIZE,
        last=True
    )

    (
        (prev_observations, prev_mask, prev_secrets, prev_envinfo),
        (observations, mask, secrets, envinfo), actions, rewards
    ) = sample

    assert (prev_secrets[:, :, 1:] == secrets[:, :, :-1]).all()
    assert (test_secrets == secrets).all()

    if has_envinfo:
        assert (prev_envinfo[:, :, 1:] == envinfo[:, :, :-1]).all()
        assert (test_envinfo == envinfo).all()
    else:
        assert prev_envinfo is None
        assert envinfo is None

    # other agent type gets secret; uncorrelated
    secrete_args = {
        'n_seeds': N_SEEDS,
        'correlated': False,
        'agent_types': ['ai' if agent_type == 'ci' else 'ci'],
        'secret_period': 1
    }
    env = NetworkGame(**config, secrete_args=secrete_args, device=device)
    env.init_episode()
    obs_shapes = env.get_observation_shapes(agent_type=agent_type)['neighbors_mask_secret_envinfo']
    has_envinfo = obs_shapes[3] is not None
    for i in range(BATCH_SIZE):
        env.init_episode()
        for j in count():
            rewards, done, info = random_step(env)
            obs, mask, secrets, envinfo = env.observe(
                agent_type=agent_type, mode='neighbors_mask_secret_envinfo')
            if done:
                break

    sample = env.sample(
        agent_type=agent_type,
        mode='neighbors_mask_secret_envinfo',
        batch_size=BATCH_SIZE,
        last=True
    )

    (
        (prev_observations, prev_mask, prev_secrets, prev_envinfo),
        (observations, mask, secrets, envinfo), actions, rewards
    ) = sample

    assert prev_secrets is None
    assert secrets is None

    warnings.warn('Sample is tested in last mode only.')
