"""
Graph Coloring Environment
"""

import random
import torch as th
import numpy as np
import yaml
import collections
from itertools import count

from aci.envs.network_game import NetworkGame


def test_neighbors(setting, agent_type, device):
    env = NetworkGame(**setting, device=device)
    n_nodes = env.n_nodes
    n_actions = env.n_actions
    ai_color = th.randint(n_actions, size=(n_nodes,))
    ci_color = th.randint(n_actions, size=(n_nodes,))
    colors = {'ai': ai_color, 'ci': ci_color}

    env.init_episode()
    rewards, done, info = env.step(colors)

    inv_mapped_colors = {'ai': ai_color[env.ai_ci_map], 'ci': ci_color}

    both_colors = th.stack([inv_mapped_colors[at] for at in env.agent_types], dim=1)

    correct = both_colors if agent_type == 'ci' else both_colors[env.ci_ai_map]
    assert (both_colors == env.state.T).all()
    assert (env.observe(agent_type=agent_type, mode='neighbors')[:, 0] == correct).all()

    obs, mask = env.observe(agent_type=agent_type, mode='neighbors_with_mask')

    test_agent = random.randint(0, n_nodes-1)

    for agent_type_idx in env.agent_types_idx.values():

        test_neighbor_colors_1 = th.masked_select(
            obs[test_agent, 1:, agent_type_idx], ~mask[test_agent, 1:])
        test_neighbor_colors_1 = th.sort(test_neighbor_colors_1)[0]

        # mapping agent from ai to ci idx
        # inverse map to because we map indices
        test_agent_ci = test_agent if agent_type == 'ci' else env.ci_ai_map[test_agent]

        # getting neighbors using ci idx
        test_neighbors_ci = env.neighbors[test_agent_ci, 1:]
        test_neighbors_ci = [t.item() for t in test_neighbors_ci if t >= 0]

        # inverse map to because we map indices
        test_neighbors = test_neighbors_ci if agent_type == 'ci' else env.ai_ci_map[test_neighbors_ci]

        test_neighbor_colors_2 = th.sort(obs[test_neighbors, 0, agent_type_idx])[0]

        assert (test_neighbor_colors_1 == test_neighbor_colors_2).all(), 'wrong neighbor observations'


def random_step(env):
    n_nodes = env.n_nodes
    n_actions = env.n_actions
    ai_color = th.randint(n_actions, size=(n_nodes,))
    ci_color = th.randint(n_actions, size=(n_nodes,))
    rewards, done, info = env.step(
        {'ai': ai_color, 'ci': ci_color}
    )
    return rewards, done, info


def test_neighbors2(setting, agent_type, device):
    env = NetworkGame(**setting, device=device)

    # test agent_types
    if (agent_type in setting['secrete_args']['agent_types']) and setting['secrete_args']['n_seeds']:
        # test correlated
        if setting['secrete_args']['correlated']:
            env.init_episode()
            rewards, done, info = random_step(env)
            obs, mask, secrets, envinfo = env.observe(
                agent_type=agent_type, mode='neighbors_mask_secret_envinfo')
            assert len(set(secrets.tolist())) == 1, 'all secrets need to be the same'

        else:
            test = False
            for i in range(1000):
                env.init_episode()
                rewards, done, info = random_step(env)
                obs, mask, secrets, envinfo = env.observe(
                    agent_type=agent_type, mode='neighbors_mask_secret_envinfo')
                if len(set(secrets.tolist())) > 1:
                    test = True
                    break
            assert test, 'secrets should differ'

        # test n_seeds
        all_secrets = []
        for i in range(1000):
            env.init_episode()
            rewards, done, info = random_step(env)
            obs, mask, secrets, envinfo = env.observe(
                agent_type=agent_type, mode='neighbors_mask_secret_envinfo')
            all_secrets.extend(secrets.tolist())
        nunique = len(set(all_secrets))
        assert nunique <= setting['secrete_args']['n_seeds']
        if setting['secrete_args']['n_seeds'] <= 10:
            assert nunique == setting['secrete_args']['n_seeds']
    else:
        env.init_episode()
        rewards, done, info = random_step(env)
        obs, mask, secrets, envinfo = env.observe(
            agent_type=agent_type, mode='neighbors_mask_secret_envinfo')
        assert secrets is None, 'for this agent type, secrets should be None'


def test_general(setting):
    device = th.device('cpu')

    env = NetworkGame(**setting, device=device)
    env.init_episode()
    n_nodes = env.n_nodes
    n_actions = env.n_actions

    # TEST: adjacency_matrix
    for i in range(n_nodes):
        for j in range(n_nodes):
            neighbors = env.neighbors[i, 1:]
            if (j == neighbors).any():
                assert env.adjacency_matrix[i, j] == True
            else:
                assert env.adjacency_matrix[i, j] == False

    # TEST: neighbors
    for i in range(env.n_nodes):
        for j in range(env.max_degree):
            neighbor = env.neighbors[i, j+1]
            if neighbor == -1:
                assert env.neighbors_mask[i, j+1] == True
            else:
                assert env.neighbors_mask[i, j+1] == False
                assert env.adjacency_matrix[i, neighbor] == True

    # TEST: ai reward
    if (setting['n_nodes'] == 6):
        ai_color = th.tensor([0, 1, 0, 1, 2, 2])[env.ci_ai_map]
        ci_color = th.tensor([2, 2, 2, 1, 2, 2])
        expected_rewards = th.tensor([0.25, 0.25, 0.25, 0.75, 0.75, 0.75])[env.ci_ai_map]
        rewards, done, info = env.step(
            {'ai': ai_color, 'ci': ci_color}
        )
        assert (rewards['ai'] == expected_rewards).all()

    # TEST: ci reward
    test_agent = random.randint(0, n_nodes-1)
    test_neighbors = env.neighbors[test_agent, 1:]
    test_neighbors = [t.item() for t in test_neighbors if t >= 0]
    ci_color = th.zeros(n_nodes, dtype=th.int64)
    ai_color = th.randint(0, n_actions, (n_nodes,), dtype=th.int64)
    ci_color[test_agent] = 1
    ci_color[test_neighbors] = 2

    rewards, done, info = env.step(
        {'ai': ai_color, 'ci': ci_color}
    )

    assert rewards['ci'][test_agent] == 1, f'Got reward of {rewards["ci"][test_agent]} instead of 1'

    ci_color = th.zeros(n_nodes, dtype=th.int64)
    rewards, done, info = env.step(
        {'ai': ai_color, 'ci': ci_color}
    )
    assert rewards['ci'][test_agent] == 0, f'Got reward of {rewards["ci"][test_agent]} instead of 0'

    # TEST: step
    env.init_episode()
    for i in count():
        ai_color = th.randint(n_actions, size=(n_nodes,))
        ci_color = th.randint(n_actions, size=(n_nodes,))
        rewards, done, info = env.step(
            {'ai': ai_color, 'ci': ci_color}
        )
        if done:
            break
    assert i == setting['episode_steps'] - 1


def test_sampling(setting, agent_type, device):
    # TEST sampling:
    env = NetworkGame(**setting, device=device)
    env.init_episode()

    n_nodes = env.n_nodes
    n_actions = env.n_actions

    batch_size = 3

    test_observations = th.empty(
        (env.n_nodes, batch_size, env.episode_steps, env.max_degree + 1, 2), dtype=th.int64)
    test_mask = th.empty((env.n_nodes, batch_size, env.episode_steps,
                          env.max_degree + 1), dtype=th.int64)
    test_actions = th.empty((env.n_nodes, batch_size, env.episode_steps), dtype=th.int64)
    test_rewards = th.empty((env.n_nodes, batch_size, env.episode_steps), dtype=th.float32)

    for i in range(batch_size):
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

    (
        (prev_observations, prev_mask), (observations, mask), actions, rewards
    ) = env.sample(agent_type=agent_type, mode='neighbors_with_mask', batch_size=batch_size, last=True)

    assert prev_observations.shape[:2] == (env.n_nodes, batch_size)
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


def test_obs_shape(setting, agent_type, device):
    env = NetworkGame(**setting, device=device)
    env.init_episode()
    rewards, done, info = random_step(env)
    obs_shapes = env.get_observation_shapes(agent_type=agent_type)

    # TODO: matrix seems to need more work
    obs_shapes.pop('matrix')

    for mode, shapes in obs_shapes.items():
        obs = env.observe(agent_type=agent_type, mode=mode)

        if not isinstance(obs, tuple):
            obs = (obs,)
            shapes = (shapes,)
        assert len(obs) == len(shapes), f'test_obs_shape, {mode} {agent_type}'
        for shape, o in zip(shapes, obs):
            # print(mode, agent_type, o.shape, shape, shapes)
            if isinstance(shape, dict):
                assert o.shape == shape['shape']
                assert o.max() <= shape['maxval']
            elif shape is None:
                assert o is None
            else:
                assert o.shape == shape

# TODO: shape test for sampleing


def test_sampling2(setting, agent_type, device):
    # TEST sampling:
    env = NetworkGame(**setting, device=device)
    env.init_episode()

    batch_size = 3

    obs_shapes = env.get_observation_shapes(agent_type=agent_type)['neighbors_mask_secret_envinfo']
    has_envinfo = obs_shapes[3] is not None
    if has_envinfo:
        envinfo_shape = obs_shapes[3]['shape'][1]
        test_envinfo = th.zeros(
            (env.n_nodes, batch_size, env.episode_steps, envinfo_shape), dtype=th.float)

    test_secrets = th.zeros((env.n_nodes, batch_size, env.episode_steps), dtype=th.int64)

    assert test_secrets.max() == 0

    for i in range(batch_size):
        env.init_episode()
        for j in count():
            rewards, done, info = random_step(env)
            obs, mask, secrets, envinfo = env.observe(
                agent_type=agent_type, mode='neighbors_mask_secret_envinfo')

            if secrets is not None:
                test_secrets[:, batch_size-i-1, j] = secrets

            if has_envinfo:
                test_envinfo[:, batch_size-i-1, j] = envinfo
            else:
                assert envinfo is None

            if done:
                break

    (
        (prev_observations, prev_mask, prev_secrets, prev_envinfo),
        (observations, mask, secrets, envinfo), actions, rewards
    ) = env.sample(agent_type=agent_type, mode='neighbors_mask_secret_envinfo', batch_size=batch_size, last=True)

    if (agent_type in setting['secrete_args']['agent_types']) and setting['secrete_args']['n_seeds'] is not None:
        assert (prev_secrets[:, :, 1:] == secrets[:, :, :-1]).all()
        assert (test_secrets == secrets).all()
    else:
        print(agent_type)
        assert test_secrets.max() == 0
        assert prev_secrets is None
        assert secrets is None

    if has_envinfo:
        assert (prev_envinfo[:, :, 1:] == envinfo[:, :, :-1]).all()
        assert (test_envinfo == envinfo).all()
    else:
        assert prev_envinfo is None
        assert envinfo is None


def test_observations(setting):
    device = th.device('cpu')
    test_obs_shape(setting, agent_type='ci', device=device)
    test_obs_shape(setting, agent_type='ai', device=device)

    test_neighbors(setting, agent_type='ci', device=device)
    test_neighbors(setting, agent_type='ai', device=device)
    test_sampling(setting, agent_type='ci', device=device)
    test_sampling(setting, agent_type='ai', device=device)

    if 'secrete_args' in setting:
        test_neighbors2(setting, agent_type='ci', device=device)
        test_neighbors2(setting, agent_type='ai', device=device)
        test_sampling2(setting, agent_type='ci', device=device)
        test_sampling2(setting, agent_type='ai', device=device)


def test_setting(setting):
    test_general(setting)
    test_observations(setting)


def test():
    print('test')

    arguments = docopt(__doc__)
    run_folder = arguments['RUN_FOLDER']

    parameter_file = os.path.join(run_folder, 'test.yml')
    out_dir = os.path.join(run_folder, 'test')

    setting = yaml.safe_load(parameter_file)
    # grid = yaml.safe_load(base_grid)
    # settings, labels = expand(setting, grid)
    # for s, l in zip(settings, labels):
    #     print(l)
    test_setting(setting)


if __name__ == "__main__":
    test()
