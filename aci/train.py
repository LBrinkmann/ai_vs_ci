"""Usage: train.py PARAMETER_FILE

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
from aci.components.scheduler import Scheduler, select_action

from aci.envs import ENVIRONMENTS
from aci.controller import CONTROLLERS
from aci.utils.io import load_yaml
from aci.utils.writer import Writer
import torch as th
import os



def run(env, controller, scheduler, writer):
    for episode in scheduler:
        writer.add_meta(_step=episode['episode'], episode=episode['episode'], mode=episode['mode'])
        print(f'Start episode {episode["episode"]}.')
        run_episode(
            env=env,
            controller=controller,
            writer=writer,
            eps=episode['eps'],
            training=episode['training'])



def run_episode(*, env, controller, eps, training, writer):
    observations = env.reset()
    agent_types = controller.keys()

    for agent_type in agent_types:
        controller[agent_type].init_episode(observations[agent_type])

    for t in count():
        # print(f'Start step {t} with test mode {test_mode}.')
        writer.add_meta(episode_step=t)

        actions = {}
        for agent_type in agent_types:
            # Select and perform an action
            proposed_action = controller[agent_type].get_q(observations[agent_type] if training[agent_type] else None)
            selected_action = select_action(proposed_actions=proposed_action, eps=eps[agent_type])
            actions[agent_type] = selected_action

        observations, rewards, done, info = env.step(actions)

        writer.add_metrics2('actions', actions, on='trace')
        writer.add_metrics2('rewards', rewards, on='trace')
        writer.add_metrics2('info', info, on='trace')

        for agent_type in agent_types:
            if training[agent_type]:
                controller[agent_type].update(
                    actions[agent_type], observations[agent_type], rewards[agent_type], done, writer=writer)
        if done:
            break


def main(*, output_path, agent_types, env_class, env_args, writer_args, meta, run_args={}, scheduler_args):
    if 'num_threads' in run_args:
        th.set_num_threads(run_args['num_threads'])

    print(f'Use {th.get_num_threads()} threads.')

    writer = Writer(output_path, **writer_args, **meta)
    # if gpu is to be used
    # device = th.device("cuda" if th.cuda.is_available() else "cpu")
    device = th.device("cpu")


    env = ENVIRONMENTS[env_class](**env_args, device=device)
    env.reset()

    scheduler = Scheduler(**scheduler_args)

    controller = {
        name: CONTROLLERS[args['controller_class']](
        observation_shape=env.observation_shape[name],
        n_actions=env.n_actions[name],
        n_agents=env.n_agents[name],
        **args['controller_args'],
        device=device)
        for name, args in agent_types.items()
    }

    run(env, controller, scheduler, writer)


if __name__ == "__main__":
    arguments = docopt(__doc__)
    parameter_file = arguments['PARAMETER_FILE']
    parameter = load_yaml(parameter_file)

    exp_dir = os.path.dirname(parameter_file)
    output_path = os.path.join(exp_dir, 'train')

    meta = {f'label.{k}': v for k, v in parameter.get('labels', {}).items()}
    main(meta=meta, output_path=output_path, **parameter['params'])
