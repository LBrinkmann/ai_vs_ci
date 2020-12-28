import torch as th
from aci.envs.network_game import NetworkGame
from .utils import random_step
import random


def _test_neighbors_v1(config, agent_type):
    device = th.device('cpu')
    env = NetworkGame(**config, device=device)
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
        test_neighbors = (
            test_neighbors_ci if agent_type == 'ci' else env.ai_ci_map[test_neighbors_ci])

        test_neighbor_colors_2 = th.sort(obs[test_neighbors, 0, agent_type_idx])[0]

        assert (test_neighbor_colors_1 == test_neighbor_colors_2).all(), \
            'wrong neighbor observations'


def _test_neighbors_v2(config, agent_type):
    device = th.device('cpu')
    N_SEEDS = 8

    # agent type gets secret; correlated
    secrete_args = {
        'n_seeds': N_SEEDS,
        'correlated': True,
        'agent_types': [agent_type],
        'secret_period': 1
    }
    env = NetworkGame(**config, secrete_args=secrete_args, device=device)
    env.init_episode()
    rewards, done, info = random_step(env)
    obs, mask, secrets, envinfo = env.observe(
        agent_type=agent_type, mode='neighbors_mask_secret_envinfo')
    assert len(set(secrets.tolist())) == 1, 'all secrets need to be the same'

    all_secrets = []
    for i in range(1000):
        env.init_episode()
        rewards, done, info = random_step(env)
        obs, mask, secrets, envinfo = env.observe(
            agent_type=agent_type, mode='neighbors_mask_secret_envinfo')
        all_secrets.extend(secrets.tolist())
    nunique = len(set(all_secrets))
    assert nunique == N_SEEDS

    # agent type gets secret; uncorrelated
    secrete_args = {
        'n_seeds': N_SEEDS,
        'correlated': False,
        'agent_types': [agent_type],
        'secret_period': 1
    }
    env = NetworkGame(**config, secrete_args=secrete_args, device=device)
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

    # other agent type gets secret
    secrete_args = {
        'n_seeds': N_SEEDS,
        'correlated': False,
        'agent_types': ['ai' if agent_type == 'ci' else 'ci'],
        'secret_period': 1
    }
    env = NetworkGame(**config, secrete_args=secrete_args, device=device)
    env.init_episode()
    rewards, done, info = random_step(env)
    obs, mask, secrets, envinfo = env.observe(
        agent_type=agent_type, mode='neighbors_mask_secret_envinfo')
    assert secrets is None, 'for this agent type, secrets should be None'
