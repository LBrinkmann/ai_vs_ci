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
from aci.components.scheduler import Scheduler, select_action

from aci.envs import ENVIRONMENTS
from aci.controller import CONTROLLERS
from aci.utils.io import load_yaml
import torch as th
import numpy as np
import os
import random


def run(envs, controller, scheduler):
    for episode in scheduler:
        print(f'Start episode {episode["episode"]} in mode {episode["mode"]}.')
        env = envs[episode['mode']]
        run_episode(
            env=env,
            controller=controller,
            **episode)


def run_episode(*, episode, env, controller, eps, training, **__):
    env.init_episode()

    # initialize episode for all controller
    for agent_type, a_controller in controller.items():
        a_controller.init_episode(episode, training[agent_type])

    for step in count():
        # collect actions of all controller here
        actions = {}
        for agent_type, a_controller in controller.items():
            # Get observations for agent type from environment
            obs = env.observe(mode=a_controller.observation_mode, agent_type=agent_type)
            # Get q values from controller
            q_values = a_controller.get_q(obs, training=training[agent_type])
            # Sample a action
            selected_action = select_action(q_values=q_values, eps=eps[agent_type])
            actions[agent_type] = selected_action

        # pass actions to environment and advance by one step
        rewards, done, info = env.step(actions)

        # allow all controller to update themself
        for agent_type, a_controller in controller.items():
            # controller do not get reward directly, but a callback to env.sample
            a_controller.update(done, env.sample, training=training[agent_type])
        if done:
            break

    env.finish_episode()


def _main(*, output_path, agent_types, env_class, env_args, meta,
          run_args={}, scheduler_args, device_name, seed=None):

    # Set seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        th.set_deterministic(True)

    # Set threads
    if 'num_threads' in run_args:
        th.set_num_threads(run_args['num_threads'])
    print(f'Use {th.get_num_threads()} threads.')

    # Set device
    device = th.device(device_name)
    print(f'Use device {device}.')

    # Create train and eval environment
    envs = {
        tm: ENVIRONMENTS[env_class](
            **env_args, device=device, out_dir=os.path.join(output_path, 'env', tm)
        )
        for tm in ['train', 'eval']
    }

    scheduler = Scheduler(**scheduler_args)
    controller = {
        name: CONTROLLERS[args['controller_class']](
            observation_shapes=envs['train'].get_observation_shapes(agent_type=name),
            n_actions=envs['train'].n_actions,
            agent_type=name,
            n_agents=envs['train'].n_nodes,
            **args['controller_args'],
            device=device)
        for name, args in agent_types.items()
    }

    run(envs, controller, scheduler)


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
