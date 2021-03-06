"""Usage: train.py RUN_FOLDER

Arguments:
    RUN_FOLDER

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


def run(envs, controller, scheduler, writer):
    for episode in scheduler:
        writer.add_meta(_step=episode['episode'], episode=episode['episode'], mode=episode['mode'])
        print(f'Start episode {episode["episode"]}.')
        env = envs[episode['mode']]
        run_episode(
            env=env,
            controller=controller,
            writer=writer,
            **episode)


def average(d):
    return {k: v.mean() for k, v in d.items()}


def run_episode(*, episode, env, controller, eps, training, writer, **__):
    env.init_episode()
    writer.add_env(env, on='env')

    agent_types = controller.keys()

    for agent_type in agent_types:
        controller[agent_type].init_episode(episode, training[agent_type])

    for t in count():
        # print(f'Start step {t} with test mode {test_mode}.')
        writer.add_meta(episode_step=t)

        actions = {}
        for agent_type in agent_types:
            # Select and perform an action
            obs = env.observe(mode=controller[agent_type].observation_mode, agent_type=agent_type)
            proposed_action = controller[agent_type].get_q(obs, training=training[agent_type])
            selected_action = select_action(proposed_actions=proposed_action, eps=eps[agent_type])
            actions[agent_type] = selected_action

        rewards, done, info = env.step(actions)

        writer.add_metrics2('actions', actions, on='trace')
        writer.add_metrics2('rewards', rewards, on='trace')
        writer.add_metrics2('info', info, on='trace')
        writer.add_metrics2('mean_rewards', average(rewards), on='mean_trace')
        writer.add_metrics2('mean_info', average(info), on='mean_trace')

        for agent_type in agent_types:
            controller[agent_type].update(done, env.sample, writer=writer, training=training[agent_type])
        if done:
            break


def _main(*, output_path, agent_types, env_class, env_args, writer_args, meta, run_args={}, scheduler_args, device_name):
    if 'num_threads' in run_args:
        th.set_num_threads(run_args['num_threads'])

    print(f'Use {th.get_num_threads()} threads.')

    writer = Writer(output_path, **writer_args, **meta)
    # if gpu is to be used
    device = th.device(device_name)
    print(device)
    # device = th.device("cpu")

    envs = {
        tm: ENVIRONMENTS[env_class](**env_args, device=device)
        for tm in ['train', 'eval']
    }

    scheduler = Scheduler(**scheduler_args)

    controller = {
        name: CONTROLLERS[args['controller_class']](
        observation_shapes=envs['train'].get_observation_shapes(agent_type=name),
        n_actions=envs['train'].n_actions,
        agent_type=name,
        n_agents=envs['train'].n_agents,
        **args['controller_args'],
        device=device)
        for name, args in agent_types.items()
    }

    run(envs, controller, scheduler, writer)


def main():
    arguments = docopt(__doc__)
    run_folder = arguments['RUN_FOLDER']

    parameter_file = os.path.join(run_folder, 'train.yml')
    out_dir = os.path.join(run_folder, 'train')
    parameter = load_yaml(parameter_file)

    meta = {f'label.{k}': v for k, v in parameter.get('labels', {}).items()}
    _main(meta=meta, output_path=out_dir, **parameter['params'])


if __name__ == "__main__":
    main()
