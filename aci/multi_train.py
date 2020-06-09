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
import sys
from aci.envs.cart import CartWrapper
from aci.envs.graph_coloring import GraphColoring
from aci.components.action_selector import MultiActionSelector
from aci.components.torch_replay_memory import ReplayMemory
from aci.controller.madqn import MADQN
from aci.ploting.training import plot_durations
from aci.ploting.screen import plot_screen
from aci.utils.io import load_yaml
import matplotlib
import matplotlib.pyplot as plt
import torch as th
from torch.utils.tensorboard import SummaryWriter

import logging
import datetime
import structlog
from structlog.stdlib import filter_by_level
from structlog.processors import JSONRenderer
from structlog.contextvars import (
    bind_contextvars,
    clear_contextvars,
    merge_contextvars
)

## logging
log = structlog.get_logger()

def add_timestamp(_, __, event_dict):
    event_dict["timestamp"] = datetime.datetime.utcnow()
    return event_dict

def parse_value(value):
    if type(value) is th.Tensor:
        if value.dim() == 0:
            return value.item()
        elif len(value) == 1:
            return value.item()
        else:
            return [i.item() for i in value]
    elif type(value) is datetime.datetime:
        return value.isoformat()
    else:
        return value

def parse_dict(_, __, event_dict):
    return {k: parse_value(v) for k,v in event_dict.items()}

def evaluation(env, controller, writer):
    bind_contextvars(mode='eval', episode_step=0)
    observations = env.reset()
    screens = [env.render()]
    episode_rewards = th.zeros(env.n_agents)

    for t in count():
        bind_contextvars(episode_step=t)

        # Select and perform an action
        actions = controller.get_action(observations)
        observations, reward, done, _ = env.step(actions)

        screens.append(env.render())
        episode_rewards += reward

        log.info(
            'finished step',
            rewards=reward, 
            avg_reward=reward.mean(), 
            episode_rewards=episode_rewards,
            avg_episode_rewards=episode_rewards.mean(),
            done=done
        )
        if done:
            writer.add_scalar('avg_reward', episode_rewards.mean())
            writer.add_scalar('episode_steps', t)
            video = th.cat(screens, dim=1)
            writer.add_video('eval_play', video, fps=1)
            break

def train_episode(env, controller, action_selector, memory, writer, batch_size):
    bind_contextvars(mode='train', episode_step=0)
    observations = env.reset()
    episode_rewards = th.zeros(env.n_agents)

    memory.push(observations)
    for t in count():
        bind_contextvars(episode_step=t, **action_selector.info())

        # Select and perform an action
        proposed_actions = controller.get_q(observations.unsqueeze(0))
        selected_action = action_selector.select_action(proposed_actions)

        observations, rewards, done, _ = env.step(selected_action[0])

        episode_rewards += rewards
        memory.push(observations, selected_action[0], rewards, done)
        log.info(
            'finished step',
            rewards=rewards,
            avg_reward=rewards.mean(), 
            episode_rewards=episode_rewards,
            avg_episode_rewards=episode_rewards.mean(),
            done=done
        )        
        if len(memory) > batch_size:
            controller.optimize(*memory.sample(batch_size))

        if done:
            writer.add_scalar('avg_reward', episode_rewards.mean())
            writer.add_scalar('episode_steps', t)
            break


def train(
        env, controller, action_selector, memory, writer, 
        num_episodes, target_update, eval_period, batch_size):
    for i_episode in range(num_episodes):
        bind_contextvars(episode=i_episode)

        log.debug('run training')
        train_episode(env, controller, action_selector, memory, writer, batch_size)

        if i_episode % target_update == 0:
            log.debug('update controller')
            controller.update()

        if i_episode % eval_period == 0:
            log.debug('run evalutation')
            evaluation(env, controller, writer)

        controller.log(writer, i_episode, i_episode % eval_period == 0)
        action_selector.log(writer, i_episode, i_episode % eval_period == 0)

envs = {
    'cart': CartWrapper,
    'graph_coloring': GraphColoring
}


def main(controller_args, env_class, selector_args, train_args, env_args, replay_memory):

    datetime_now = datetime.datetime.utcnow().isoformat()
    logfile = f"logs/{datetime_now}.log"
    tb_dir = f"tensorboard/{datetime_now}"

    logging.basicConfig(
        handlers=[logging.FileHandler(logfile)], # logging.StreamHandler()], 
        format="%(message)s",
        level='INFO'
    )
    structlog.configure(
        processors=[
            filter_by_level,
            merge_contextvars,
            add_timestamp,
            parse_dict,
            JSONRenderer(sort_keys=True)
        ],
        # context_class=structlog.threadlocal.wrap_dict(dict),
        logger_factory=structlog.stdlib.LoggerFactory(),
    )

    # if gpu is to be used
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    writer = SummaryWriter(log_dir=tb_dir)

    env = envs[env_class](**env_args, device=device)
    env.reset()
    n_actions = env.n_actions
    observation_shape = env.observation_shape
    memory = ReplayMemory(replay_memory)

    controller = MADQN(
        observation_shape=observation_shape,
        n_actions=n_actions,
        n_agents=env.n_agents,
        **controller_args,
        device=device)

    action_selector = MultiActionSelector(device=device, **selector_args)
    train(env, controller, action_selector, memory, writer, **train_args)
    log.info('complete')


if __name__ == "__main__":
    arguments = docopt(__doc__)
    parameter = load_yaml(arguments['PARAMETER_YAML'])
    main(**parameter)
