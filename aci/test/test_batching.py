from aci.train import get_environment, get_observer
import yaml
import torch as th

settings = """
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

BATCH_SIZE = 100
DEVICE = th.device("cpu")


def random_action(n_nodes, n_agent_types, n_actions, **_):
    return th.randint(0, n_actions, (n_nodes, n_agent_types))


def test_sampling():
    settings = yaml.safe_load(settings)

    env = get_environment(**settings['environment_args'], device=DEVICE)
    env.init_episode()
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
        env.init_episode()
        for j in count():
            actions = random_action(env.info)

            state, rewards, done = env.step(actions)

            test_states.append(state)
            test_actions.append(actions)
            test_rewards.append(test_rewards)
            test_observations.append(
                {
                    name: obs(**state)
                    for name, obs in observer.items()
                }
            )

    for atidx, (name, obs) in enumerate(observer.items()):
        states, actions, rewards = env.sample(
            batch_size=BATCH_SIZE, last=True, agent_type=name)

        observations = obs(states)

        for i in range(BATCH_SIZE):
            for k, v in states.items():
                assert th.allclose(v, test_states[i][k])
            for k, v in observations.items():
                assert th.allclose(v, test_states[i][name][k])
            assert th.allclose(actions, test_actions[i][atidx])
            assert th.allclose(rewards, test_rewards[i][atidx])


if __name__ == "__main__":
    test_sampling()
