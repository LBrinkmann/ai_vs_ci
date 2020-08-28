"""
Graph Coloring Environment
"""

import random
import torch as th
import numpy as np
import yaml
import collections
from itertools import count

from aci.envs.adversial_graph_coloring_historized import AdGraphColoringHist


# Tests


base_settings = """
n_agents: 6
n_actions: 3
network_period: 1
mapping_period: 1
max_history: 50
graph_args:
    constrains: 
        max_max_degree: 4
        connected: True
    graph_args:
        p: 0.2
    graph_type: erdos_renyi
episode_length: 50
rewards_args:
    ai:
        avg_catch: 0.5
        avg_coordination: 0
        ind_catch: 0.5
        ind_coordination: 0
    ci:
        avg_catch: 0
        avg_coordination: 0
        ind_catch: 0
        ind_coordination: 1
"""
base_grid = """
n_agents: [6, 20]
network_period: [0, 1, 2]
mapping_period: [0, 1, 2]
"""

def expand(setting, grid):
    settings = [setting]
    labels = [{}]
    for key, values in grid.items():
        settings = [{**s, key: v} for s in settings for v in values]
        labels = [{**l, key: v} for l in labels for v in values]
    return settings, labels



def test_neighbors(setting, agent_type, device):
    env = AdGraphColoringHist(**setting, device=device)
    n_agents = env.n_agents
    n_actions = env.n_actions
    ai_color = th.randint(n_actions, size=(n_agents,))
    ci_color = th.randint(n_actions, size=(n_agents,))
    env.init_episode()
    rewards, done, info = env.step(
        {'ai': ai_color, 'ci': ci_color}
    )

    correct = ci_color if agent_type == 'ci' else ci_color[env.ci_ai_map]
    assert (env.observe(agent_type=agent_type, mode='neighbors', with_mask=False)[:,0] == correct).all()

    obs, mask = env.observe(agent_type=agent_type, mode='neighbors', with_mask=True)

    test_agent = random.randint(0, n_agents-1)

    test_neighbor_colors_1 = th.masked_select(obs[test_agent,1:], ~mask[test_agent,1:])
    test_neighbor_colors_1 = th.sort(test_neighbor_colors_1)[0]
    
    # mapping agent from ai to ci idx
    test_agent_ci = test_agent if agent_type == 'ci' else env.ci_ai_map[test_agent]  # inverse map to because we map indices

    # getting neighbors using ci idx
    test_neighbors_ci = env.neighbors[test_agent_ci, 1:]
    test_neighbors_ci = [t.item() for t in test_neighbors_ci if t >= 0]

    test_neighbors_ai = test_neighbors_ci if agent_type == 'ci' else env.ai_ci_map[test_neighbors_ci] # inverse map to because we map indices

    test_neighbor_colors_2 = th.sort(obs[test_neighbors_ai,0])[0]

    assert (test_neighbor_colors_1 == test_neighbor_colors_2).all(), 'wrong neighbor observations'


def test_general(setting):
    device = th.device('cpu')
    env = AdGraphColoringHist(**setting, device=device)
    env.init_episode()
    n_agents = env.n_agents
    n_actions = env.n_actions

    # TEST: adjacency_matrix
    for i in range(n_agents):
        for j in range(n_agents):
            neighbors = env.neighbors[i, 1:]
            if (j == neighbors).any():
                assert env.adjacency_matrix[i,j] == True
            else:
                assert env.adjacency_matrix[i,j] == False

    # TEST: neighbors
    for i in range(env.n_agents):
        for j in range(env.max_degree):
            neighbor = env.neighbors[i, j+1]
            if neighbor == -1:
                assert env.neighbors_mask[i,j+1] == True
            else:
                assert env.neighbors_mask[i,j+1] == False
                assert env.adjacency_matrix[i,neighbor] == True

    # TEST: ai reward
    if (setting['n_agents'] == 6):
        ai_color = th.tensor([0,1,0,1,2,2])[env.ci_ai_map]
        ci_color = th.tensor([2,2,2,1,2,2])
        expected_rewards = th.tensor([0.25,0.25,0.25, 0.75,0.75,0.75])[env.ci_ai_map]
        rewards, done, info = env.step(
            {'ai': ai_color, 'ci': ci_color}
        )
        assert (rewards['ai'] ==  expected_rewards).all()

    # TEST: ci reward
    test_agent = random.randint(0, n_agents-1)
    test_neighbors = env.neighbors[test_agent, 1:]
    test_neighbors = [t.item() for t in test_neighbors if t >= 0]
    ci_color = th.zeros(n_agents, dtype=th.int64)
    ai_color = th.randint(0, n_actions, (n_agents,), dtype=th.int64)
    ci_color[test_agent] = 1
    ci_color[test_neighbors] = 2

    rewards, done, info = env.step(
        {'ai': ai_color, 'ci': ci_color}
    )
    assert rewards['ci'][test_agent] == 1, f'Got reward of {rewards["ci"][test_agent]} instead of 1'

    ci_color = th.zeros(n_agents, dtype=th.int64)
    rewards, done, info = env.step(
        {'ai': ai_color, 'ci': ci_color}
    )
    assert rewards['ci'][test_agent] == 0, f'Got reward of {rewards["ci"][test_agent]} instead of 0'


    # TEST: step
    env.init_episode()
    for i in count():
        ai_color = th.randint(n_actions, size=(n_agents,))
        ci_color = th.randint(n_actions, size=(n_agents,))
        rewards, done, info = env.step(
            {'ai': ai_color, 'ci': ci_color}
        )
        if done:
            break
    assert i == setting['episode_length'] - 1


def test_sampling(setting, agent_type, device):
    # TEST sampling:
    env = AdGraphColoringHist(**setting, device=device)
    env.init_episode()

    n_agents = env.n_agents
    n_actions = env.n_actions

    batch_size = 3

    test_observations = th.empty((env.n_agents, batch_size, env.episode_length, env.max_degree + 1), dtype=th.int64)
    test_mask = th.empty((env.n_agents, batch_size, env.episode_length, env.max_degree + 1), dtype=th.int64)
    test_actions = th.empty((env.n_agents, batch_size, env.episode_length), dtype=th.int64)
    test_rewards = th.empty((env.n_agents, batch_size, env.episode_length), dtype=th.float32)

    for i in range(batch_size):
        env.init_episode()
        for j in count():
            ai_color = th.randint(n_actions, size=(n_agents,))
            ci_color = th.randint(n_actions, size=(n_agents,))
            actions = {'ai': ai_color, 'ci': ci_color}
            rewards, done, info = env.step(actions)
            obs, mask = env.observe(agent_type=agent_type, mode='neighbors', with_mask=True)
            test_observations[:,2-i,j] = obs
            test_mask[:,2-i,j] = mask
            test_actions[:,2-i,j] = actions[agent_type]
            test_rewards[:,2-i,j] = rewards[agent_type]

            if done:
                break

    (
        prev_observations, observations, prev_mask, mask, actions, rewards
    ) = env.sample(agent_type=agent_type, mode='neighbors', batch_size=batch_size, last=True, with_mask=True)

    assert prev_observations.shape[:2] == (env.n_agents, batch_size)
    assert (
        rewards.shape[:3] == actions.shape[:3] ==
        prev_observations.shape[:3] == observations.shape[:3] ==
        prev_mask.shape[:3] == mask.shape[:3]
    )
    assert (prev_observations[:,:,1:] == observations[:,:,:-1]).all()
    assert (prev_mask[:,:,1:] == mask[:,:,:-1]).all()

    # for ci only
    if agent_type == 'ci':
        assert (observations[:,:,:,0] == actions).all()

    assert (test_observations == observations).all()
    assert (test_rewards == rewards).all()
    assert (test_actions == actions).all()
    assert (test_mask == mask).all()



def test_observations(setting):
    device = th.device('cpu')
    test_neighbors(setting, agent_type='ci', device=device)
    test_neighbors(setting, agent_type='ai', device=device)

    test_sampling(setting, agent_type='ci', device=device)
    test_sampling(setting, agent_type='ai', device=device)



def test_setting(setting):
    test_general(setting)
    test_observations(setting)



def test():
    setting = yaml.safe_load(base_settings)
    grid = yaml.safe_load(base_grid)
    settings, labels = expand(setting, grid)
    for s, l in zip(settings, labels):
        print(l)
        test_setting(s)


    # test_setting(setting)



if __name__ == "__main__":
    test()    