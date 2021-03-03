"""Usage: train.py RUN_FOLDER

Arguments:
    RUN_FOLDER

Outputs:
    ....
"""

from itertools import count
from docopt import docopt
from aci.scheduler.scheduler import Scheduler
from aci.envs import ENVIRONMENTS
from aci.controller import CONTROLLERS
from aci.observer import OBSERVER
from aci.controller.selector import eps_greedy

from aci.utils.io import load_yaml
from aci.utils.random import set_seeds
import torch as th
import numpy as np
import os


def run(envs, controller, observer, scheduler, device):
    for episode in scheduler:
        print(f'Start episode {episode["episode"]} in mode {episode["mode"]}.')
        env = envs[episode['mode']]
        run_episode(
            env=env,
            controller=controller,
            observer=observer,
            device=device,
            **episode)


def run_episode(*, episode, env, controller, observer, eps, training, device, **__):
    actions, state, rewards = env.init_episode()

    # initialize episode for all controller
    observations = {}
    for agent_type, a_controller in controller.items():
        observations[agent_type] = observer[agent_type](**state)
        for k, v in observations[agent_type].items():
            print(k, v.device)
        a_controller.init_episode(
            observation=observations[agent_type], action=actions[agent_type],
            reward=rewards[agent_type], episode=episode)

    for step in count():
        # collect actions of all controller here

        for agent_type, a_controller in controller.items():
            # Get q values from controller
            q_values = a_controller.get_q(**observations[agent_type])

            # Sample a action
            selected_action = eps_greedy(q_values=q_values, eps=eps[agent_type], device=device)
            actions[agent_type] = selected_action

        # pass actions to environment and advance by one step
        state, rewards, done = env.step(actions)

        for agent_type, a_controller in controller.items():
            observations[agent_type] = observer[agent_type](**state)

            # allow controller to update
            if training[agent_type]:
                a_controller.update(
                    observations[agent_type], actions[agent_type], rewards[agent_type], done,
                    step=step, episode=episode)
        if done:
            break

    env.finish_episode()


def get_environment(class_name, **kwargs):
    return ENVIRONMENTS[class_name](**kwargs)


def get_observer(class_name, **kwargs):
    return OBSERVER[class_name](**kwargs)


def get_controller(class_name, **kwargs):
    return CONTROLLERS[class_name](**kwargs)


def _main(*, output_path, environment_args, observer_args, controller_args, meta,
          run_args={}, scheduler_args, save_interval, device_name, seed=None):

    set_seeds(seed)

    # Set threads
    if 'num_threads' in run_args:
        th.set_num_threads(run_args['num_threads'])
    print(f'Use {th.get_num_threads()} threads.')

    # Set device
    device = th.device(device_name)
    print(f'Use device {device}.')

    # Create train and eval environment
    envs = {
        tm: get_environment(
            **environment_args, device=device, out_dir=os.path.join(output_path, 'env', tm),
            save_interval=save_interval[tm]
        )
        for tm in ['train', 'eval']
    }

    scheduler = Scheduler(**scheduler_args)
    observer = {
        name: get_observer(
            env_info=envs['train'].info,
            agent_type=name,
            **args,
            device=device)
        for name, args in observer_args.items()
    }

    controller = {
        name: get_controller(
            observation_shape=observer[name].shape,
            env_info=envs['train'].info,
            agent_type=name,
            **args,
            device=device)
        for name, args in controller_args.items()
    }

    run(envs, controller, observer, scheduler, device)


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
