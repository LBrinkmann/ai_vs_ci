"""Usage: train.py PARAMETER_FILE OUTPUT_PATH

Arguments:
    RUN_PATH

Outputs:
    ....
"""


# Taken from:
# https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py

from itertools import count
from docopt import docopt
import sys
from aci.components.action_selector import MultiActionSelector

from aci.envs import ENVIRONMENTS
from aci.controller import CONTROLLERS
from aci.utils.io import load_yaml
from aci.utils.writer import Writer
import torch as th


def calc_metrics(rewards, episode_rewards):
    return {
        'rewards': rewards,
        'avg_reward': rewards.mean(),
        'episode_rewards': episode_rewards,
        'avg_episode_rewards': episode_rewards.mean()
    }


def run_episode(*, env, controller, action_selector, writer, test_mode=False):
    observations = env.reset()
    agent_types = controller.keys()

    # added
    if not test_mode:
        for agent_type in agent_types:
            controller[agent_type].init_episode(observations[agent_type])

    for t in count():
        writer.add_meta(episode_step=t)

        for agent_type in agent_types:
            ainfo = {f"{agent_type}_{k}": v for k,v in action_selector[agent_type].info().items()}
            writer.add_meta(**ainfo)

        actions = {}
        for agent_type in agent_types:
            # Select and perform an action
            this_obs = observations[agent_type]
            proposed_actions = controller[agent_type].get_q(this_obs)
            selected_action = action_selector[agent_type].select_action(
                proposed_actions, test_mode=test_mode)
            actions[agent_type] = selected_action

        observations, rewards, done, _ = env.step(actions, writer)

        if not test_mode:
            for agent_type in agent_types:
                controller[agent_type].update(
                    actions[agent_type], observations[agent_type], rewards[agent_type], done, writer=writer)
        if done:
            break

def train(
        env, controller, action_selector, writer,  num_episodes, eval_period):
    for i_episode in range(num_episodes):
        writer.add_meta(_step=i_episode, episode=i_episode, mode='train')

        run_episode(
            env=env,
            controller=controller,
            action_selector=action_selector,
            writer=writer)

        if i_episode % eval_period == 0:
            writer.add_meta(mode='eval')
            run_episode(
                env=env,
                controller=controller,
                action_selector=action_selector,
                writer=writer,
                test_mode=True
            )


def main(agent_types, env_class, train_args, env_args, writer_args, meta):
    writer = Writer(output_path, **writer_args, **meta)
    # if gpu is to be used
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    env = ENVIRONMENTS[env_class](**env_args, device=device)
    env.reset()

    controller = {
        name: CONTROLLERS[args['controller_class']](
        observation_shape=env.observation_shape[name],
        n_actions=env.n_actions[name],
        n_agents=env.n_agents[name],
        **args['controller_args'],
        device=device)
        for name, args in agent_types.items()
    }

    action_selector = {
        name: MultiActionSelector(device=device, **args['selector_args'])
        for name, args in agent_types.items()
    }

    train(env, controller, action_selector, writer, **train_args)


if __name__ == "__main__":
    arguments = docopt(__doc__)
    parameter_file = arguments['PARAMETER_FILE']
    output_path = arguments['OUTPUT_PATH']
    parameter = load_yaml(parameter_file)

    meta = {f'label.{k}': v for k, v in parameter.get('labels', {}).items()}
    main(meta=meta, **parameter['params'])
