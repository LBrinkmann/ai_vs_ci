import torch as th
from aci.envs.network_game import create_maps, create_control


def test_create_maps():
    n_nodes = 1000
    agent_type_args = {
        'ci': {'mapping_type': 'random'},
        'ai': {'mapping_type': 'fixed'}
    }
    mapping = create_maps(agent_type_args, n_nodes)
    assert mapping.shape == (n_nodes, len(agent_type_args))
    assert (mapping[:, 1] == th.arange(n_nodes)).all()
    assert (mapping[:, 0] != th.arange(n_nodes)).any()


def test_control():
    device = th.device('cpu')
    n_nodes = 10000
    n_agent_types = 2
    n_control = 5

    correlated = True
    cross_correlated = True
    control = create_control(n_nodes, n_agent_types, n_control,
                             correlated, cross_correlated, device)
    control = control.squeeze(0).squeeze(0)
    assert control.shape == (n_nodes, n_agent_types)
    assert (control[:, 0] == control[:, 1]).all()
    assert (control[th.randint(0, n_nodes, (n_nodes,)), 0] ==
            control[th.randint(0, n_nodes, (n_nodes,)), 0]).all()

    correlated = False
    cross_correlated = True
    control = create_control(n_nodes, n_agent_types, n_control,
                             correlated, cross_correlated, device)
    control = control.squeeze(0).squeeze(0)
    assert (control[:, 0] == control[:, 1]).all()
    assert (control[th.randint(0, n_nodes, (n_nodes,)), 0] !=
            control[th.randint(0, n_nodes, (n_nodes,)), 0]).any()

    correlated = True
    cross_correlated = False
    control = create_control(n_nodes, n_agent_types, n_control,
                             correlated, cross_correlated, device)
    control = control.squeeze(0).squeeze(0)
    assert (control[:, 0] != control[:, 1]).any()
    assert (control[th.randint(0, n_nodes, (n_nodes,)), 0] ==
            control[th.randint(0, n_nodes, (n_nodes,)), 0]).all()

    correlated = False
    cross_correlated = False
    control = create_control(n_nodes, n_agent_types, n_control,
                             correlated, cross_correlated, device)
    control = control.squeeze(0).squeeze(0)
    assert control.max() == (n_control - 1)
    assert (control[:, 0] != control[:, 1]).any()
    assert (control[th.randint(0, n_nodes, (n_nodes,)), 0] !=
            control[th.randint(0, n_nodes, (n_nodes,)), 0]).any()


if __name__ == "__main__":
    test_create_maps()
    test_control()
