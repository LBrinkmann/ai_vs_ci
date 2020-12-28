import torch as th
from aci.envs.network_game import NetworkGame


def general_test(config):
    device = th.device('cpu')
    env = NetworkGame(**config, device=device)
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
        for j in range(env.max_neighbors):
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
