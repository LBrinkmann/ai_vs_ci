from aci.train import get_environment, get_observer
import yaml
import torch as th
from itertools import count

SETTINGS = """
environment_args:
    class_name: network_game
    n_nodes: 20
    n_actions: 4
    episode_steps: 20
    max_history: 50
    network_period: 2
    mapping_period: 5
    graph_args:
        constrains:
            max_max_degree: 8
            connected: True
        graph_args:
            p: 0.2
        graph_type: erdos_renyi
    reward_args:
        ci:
            local_coordination:
                ci: 2
        ai:
            ind_crosscoordination:
                ci: -3
    agent_type_args:
        ci:
            mapping_type: fixed
        ai:
            mapping_type: random
    control_args:
        correlated: True
        cross_correlated: True
        n_control: 8
observer_args:
    ci:
        class_name: neighbor
        neighbor_view_args:
            actions_view: ["ci"]
        control_view_args:
            metric_view:
                -   agent_type: ci
                    metric_name: global_coordination
                -   agent_type: ci
                    metric_name: local_coordination
            control_view: "ci"
    ai:
        class_name: neighbor
        neighbor_view_args:
            actions_view: ["ci"]
"""

BATCH_SIZE = 10
DEVICE = th.device("cpu")


def random_action(n_nodes, n_agent_types, n_actions, **_):
    return th.randint(0, n_actions, (1, 1, n_nodes, n_agent_types), dtype=th.int64, device=DEVICE)


def test_sampling():
    settings = yaml.safe_load(SETTINGS)

    env = get_environment(**settings['environment_args'], device=DEVICE)
    observer = {
        name: get_observer(
            env_info=env.info,
            agent_type=name,
            **args,
            device=DEVICE)
        for name, args in settings['observer_args'].items()
    }
    test_states = []
    test_rewards = []
    test_actions = []
    test_observations = []

    for i in range(BATCH_SIZE):
        print(env.episode)
        state, rewards, done = env.init_episode()
        _test_states = [state]
        _test_rewards = []
        _test_actions = []
        _test_observations = [{
            name: obs(**state)
            for name, obs in observer.items()
        }]
        for j in count():
            actions = random_action(**env.info)

            state, rewards, done = env.step(actions)

            _test_states.append(state)
            _test_actions.append(actions)
            _test_rewards.append(rewards)
            _test_observations.append(
                {
                    name: obs(**state)
                    for name, obs in observer.items()
                }
            )
            if done:
                break
        test_states.append(_test_states)
        test_rewards.append(_test_rewards)
        test_actions.append(_test_actions)
        test_observations.append(_test_observations)

    for atidx, (name, obs) in enumerate(observer.items()):
        states, actions, rewards = env.sample(
            batch_size=BATCH_SIZE, last=True, agent_type=name)

        observations = obs(**states)

        for i in range(BATCH_SIZE):
            sample_idx = BATCH_SIZE - i - 1

            for j in range(env.info['episode_steps']):
                assert th.allclose(actions[sample_idx, j], test_actions[i][j][0, 0, :, atidx])
                assert th.allclose(rewards[sample_idx, j], test_rewards[i][j][0, 0, :, atidx])
                for k in ['neighbors', 'neighbors_mask', 'agent_map']:
                    assert (states[k][sample_idx] == test_states[i][j][k]).all()
                for k in ['actions', 'metrics', 'control_int']:
                    assert (states[k][sample_idx, j] == test_states[i][j][k]).all()
                for k, v in observations.items():
                    if v is None:
                        assert test_observations[i][j][name][k] is None
                    else:
                        assert (v[sample_idx, j] == test_observations[i][j][name][k]).all()


if __name__ == "__main__":
    test_sampling()
