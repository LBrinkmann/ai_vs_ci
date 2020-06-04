"""Usage: train.py PARAMETER_YAML

Arguments:
    PARAMETER_YAML        A yaml parameter file.

Outputs:
    ....
"""


# Taken from:
# https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py

from itertools import count
from docopt import docopt
from aci.envs.graph_coloring import GraphColoring
from aci.controller.action_selector import MultiActionSelector
from aci.controller.graph import MADQN
from aci.ploting.training import plot_durations
from aci.ploting.screen import plot_screen
from aci.utils.io import load_yaml
import matplotlib
import matplotlib.pyplot as plt
import torch as th
from torch.utils.tensorboard import SummaryWriter

import logging
import datetime
from structlog import wrap_logger
from structlog.processors import JSONRenderer
from structlog.contextvars import (
    bind_contextvars,
    clear_contextvars
)

## logging

logging.basicConfig(
    handlers=[logging.FileHandler("example1.log"), logging.StreamHandler()], 
    format="%(message)s"
)

def add_timestamp(_, __, event_dict):
    event_dict["timestamp"] = datetime.datetime.utcnow()
    return event_dict

log = wrap_logger(
    logging.getLogger(__name__),
    processors=[
        add_timestamp,
        JSONRenderer(indent=1, sort_keys=True)
    ]
)


def evaluation(env, controller, writer):
    bind_contextvars(mode='train', episode_step=0)
    observations = env.reset()
    screens = [observations]
    episode_rewards = th.zeros(env.n_agents)

    for t in count():
        bind_contextvars(episode_step=t)

        # Select and perform an action
        actions = controller.get_action(observations)
        observations, reward, done, _ = env.step(actions)

        screens.append(observations)
        episode_rewards += reward

        log.info(
            'finished step',
            rewards=reward.cpu().numpy(), 
            avg_reward=reward.mean().cpu().item(), 
            episode_rewards=episode_rewards.cpu().numpy(),
            avg_episode_rewards=episode_rewards.mean().cpu().item(),
            done=done
        )
        if done:
            writer.add_scalars('avg_reward', episode_rewards.mean())
            writer.add_scalars('episode_steps', t)
            video = th.stack(screens).unsqueeze(1).unsqueeze(0).repeat(1,1,3,1,1)
            writer.add_video('eval_play', video, fps=1)

def train_episode(env, controller, action_selector, writer, target_update):
    bind_contextvars(mode='train', episode_step=0)
    observations = env.reset()
    episode_rewards = th.zeros(env.n_agents)

    last_observations = observations
    for t in count():
        bind_contextvars(episode_step=t, **action_selector.info())

        # Select and perform an action
        proposed_actions = controller.get_action(observations)
        selected_action = action_selector.select_action(proposed_actions)
        observations, reward, done, _ = env.step(selected_action)

        episode_rewards += reward

        controller.push_transition(last_observations, selected_action, observations, reward)
        last_observations = observations
        controller.optimize()

        log.info(
            'finished step',
            rewards=reward.cpu().numpy(), 
            avg_reward=reward.mean().cpu().item(), 
            episode_rewards=episode_rewards.cpu().numpy(),
            avg_episode_rewards=episode_rewards.mean().cpu().item(),
            done=done
        )

        if done:
            writer.add_scalars('avg_reward', episode_rewards.mean())
            writer.add_scalars('episode_steps', t)
            break


def train(env, controller, action_selector, writer, num_episodes, target_update, eval_period):
    for i_episode in range(num_episodes):
        bind_contextvars(episode=i_episode)

        log.debug('run training')
        train_episode(env, controller, action_selector, writer, target_update)

        if i_episode % target_update == 0:
            log.debug('update controller')
            controller.update()

        if i_episode % eval_period == 0:
            log.debug('run evalutation')
            evaluation(env, controller, writer)

        controller.log(writer, i_episode, i_episode % eval_period == 0)
        action_selector.log(writer, i_episode, i_episode % eval_period == 0)


def main(opt_args, selector_args, train_args, env_args, replay_memory):
    # if gpu is to be used
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    writer = SummaryWriter()
    env = GraphColoring(**env_args, device=device)
    env.reset()
    n_actions = env.n_actions
    observation_shape = env.observation_shape

    controller = MADQN(
        observation_shape=observation_shape,
        n_actions=n_actions,
        n_agents=env.n_agents,
        opt_args=opt_args, 
        replay_memory=replay_memory,
        device=device)

    action_selector = MultiActionSelector(device=device, **selector_args)
    train(env, controller, action_selector, writer, **train_args)
    log.info('complete')


if __name__ == "__main__":
    arguments = docopt(__doc__)
    parameter = load_yaml(arguments['PARAMETER_YAML'])
    main(**parameter)
