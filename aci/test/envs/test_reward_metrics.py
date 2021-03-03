import torch as th
import random
from aci.envs.reward_metrics import project_on_neighbors, calc_metrics, calc_reward, create_reward_vec, METRIC_NAMES


NEIGHBOR = th.tensor([
    [0, 1, -1],
    [1, 0, -1],
    [2, 3, 4],
    [3, 2, -1],
    [4, 2, -1]
]).unsqueeze(0)

MASK = NEIGHBOR == -1

ACTIONS = th.tensor([
    [0, 0], [1, 1], [2, 1], [2, 2], [2, 2]
]).unsqueeze(0).unsqueeze(0)

H, S, P, T = ACTIONS.shape
N = NEIGHBOR.shape[-1]


DEVICE = th.device("cpu")


def get_m_idx(name):
    return METRIC_NAMES.index(name)


def test_project_on_neighbors():
    test_shapes = [[2, 4], [2]]
    for ts in test_shapes:
        tensor = th.rand((H, S, P, *ts))
        fill_value = 0.4
        p_tensor = project_on_neighbors(tensor, NEIGHBOR, MASK, fill_value=fill_value)
        for i in range(10):

            test_node = th.randint(0, P, (1,))[0]
            test_neighbor = th.randint(0, N, (1,))[0]
            test_neighbor_node = NEIGHBOR[0, test_node, test_neighbor]

            if test_neighbor_node == -1:
                assert (p_tensor[0, 0, test_node, test_neighbor] == fill_value).all()
            else:
                assert (p_tensor[0, 0, test_node, test_neighbor]
                        == tensor[0, 0, test_neighbor_node]).all()


def test_calc_metrics():
    # references
    ind_crosscoordination = [1, 1, 0, 1, 1]
    local_crosscoordination = [1, 1, 2/3, 1/2, 1/2]
    global_crosscoordination = [4/5, 4/5, 4/5, 4/5, 4/5]

    ind_coordination = [[0, 0], [0, 0], [1, 0], [1, 0], [1, 0]]
    local_coordination = [[0, 0], [0, 0], [1, 0], [1, 0], [1, 0]]
    global_coordination = [[3/5, 0], [3/5, 0], [3/5, 0], [3/5, 0], [3/5, 0]]

    ind_anticoordination = [[1, 1], [1, 1], [0, 1], [0, 1], [0, 1]]
    local_anticoordination = [[1, 1], [1, 1], [0, 1], [0, 1], [0, 1]]
    global_anticoordination = [[2/5, 1], [2/5, 1], [2/5, 1], [2/5, 1], [2/5, 1]]

    metrics = calc_metrics(ACTIONS, NEIGHBOR, MASK)
    metrics = metrics.squeeze(0).squeeze(0)

    assert (metrics[:, 0, get_m_idx('ind_crosscoordination')] ==
            metrics[:, 1, get_m_idx('ind_crosscoordination')]).all()

    assert (metrics[:, 0, get_m_idx('ind_crosscoordination')]
            == th.tensor(ind_crosscoordination)).all()
    assert (metrics[:, 0, get_m_idx('local_crosscoordination')]
            == th.tensor(local_crosscoordination)).all()
    assert th.allclose(metrics[:, 0, get_m_idx('global_crosscoordination')],
                       th.tensor(global_crosscoordination))

    assert (metrics[:, :, get_m_idx('ind_coordination')]
            == th.tensor(ind_coordination)).all()
    assert (metrics[:, :, get_m_idx('local_coordination')]
            == th.tensor(local_coordination)).all()
    assert (metrics[:, :, get_m_idx('global_coordination')]
            == th.tensor(global_coordination)).all()

    assert (metrics[:, :, get_m_idx('ind_anticoordination')]
            == th.tensor(ind_anticoordination)).all()
    assert (metrics[:, :, get_m_idx('local_anticoordination')]
            == th.tensor(local_anticoordination)).all()
    assert (metrics[:, :, get_m_idx('global_anticoordination')]
            == th.tensor(global_anticoordination)).all()


def test_create_reward_vec():
    agent_type_args = {
        'ci':  {'local_coordination': {'ci': 2}},
        'ai':  {'ind_crosscoordination': {'ci': -3}},
    }
    ref = th.zeros((2, 9, 2), dtype=th.float)
    ref[0, get_m_idx('local_coordination'), 0] = 2
    ref[1, get_m_idx('ind_crosscoordination'), 0] = -3

    reward_vec = create_reward_vec(agent_type_args, DEVICE)

    assert th.allclose(ref, reward_vec)


def test_calc_reward():
    agent_type_args = {
        'ci':  {'local_coordination': {'ci': 2}},
        'ai':  {'ind_crosscoordination': {'ci': -3}},
    }
    reward_vec = create_reward_vec(agent_type_args, DEVICE)
    metrics = calc_metrics(ACTIONS, NEIGHBOR, MASK)
    reward = calc_reward(metrics, reward_vec).squeeze(0).squeeze(0)  # h s+ p t

    _metrics = metrics.squeeze(0).squeeze(0)
    ci_reward = _metrics[:, 0, get_m_idx('local_coordination')] * 2
    ai_reward = _metrics[:, 0, get_m_idx('ind_crosscoordination')] * -3

    assert th.allclose(reward[:, 0], ci_reward)
    assert th.allclose(reward[:, 1], ai_reward)


if __name__ == "__main__":
    test_project_on_neighbors()
    test_calc_metrics()
    test_create_reward_vec()
    test_calc_reward()
