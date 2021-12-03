import torch as th
from aci.observer.neighbor import NeighborView, ViewEncoder
from aci.envs.reward_metrics import calc_metrics, METRIC_NAMES

NEIGHBOR = th.tensor([
    [0, 1, -1],
    [1, 0, -1],
    [2, 3, -1],
    [3, 2, -1]
]).unsqueeze(0)

MASK = NEIGHBOR == -1

ACTIONS = th.tensor([
    [0, 0], [1, 0], [2, 1], [2, 2]
]).unsqueeze(0).unsqueeze(0)

CONTROL_INT = th.tensor([
    [3, 0], [3, 1], [3, 2], [3, 3]
]).unsqueeze(0).unsqueeze(0)

METRICS = calc_metrics(ACTIONS, NEIGHBOR, MASK)

H, S, P, T = ACTIONS.shape
N = NEIGHBOR.shape[-1]

DEVICE = th.device('cpu')
ENV_INFO = {
    'n_nodes': P,
    'n_actions': 3,
    'n_control': 4,
    'agent_types': ['ci', 'ai'],
    'metric_names': METRIC_NAMES
}


def test_view_encoder():
    state = {
        'actions': ACTIONS,
        'control_int': CONTROL_INT,
        'metrics': METRICS
    }

    print(ENV_INFO)

    settings = {"actions_view": ['ci'], 'metric_view': [
        {'agent_type': 'ci', 'metric_name': 'ind_coordination'}],
        "control_view": 'ai'}
    v_enc = ViewEncoder(**settings, env_info=ENV_INFO, device=DEVICE)
    test = v_enc(**state).squeeze(0).squeeze(0)
    assert test.shape[-1] == (ENV_INFO['n_actions'] + 1 + 2)

    for i in range(10):
        test_node = th.randint(0, ENV_INFO['n_nodes'], (1,))[0]
        test_action = th.randint(0, ENV_INFO['n_actions'], (1,))[0]
        # import ipdb
        # ipdb.set_trace()
        assert test[test_node, test_action] == (ACTIONS[0, 0, test_node, 0] == test_action)

    assert th.allclose(test[:, ENV_INFO['n_actions']], th.tensor([0., 0., 1., 1.]))

    ref_control = th.tensor([[0., 0], [0, 1], [1, 0], [1, 1]])
    assert th.allclose(test[:, -2:], ref_control)


if __name__ == "__main__":
    test_view_encoder()
