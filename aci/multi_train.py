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
from aci.envs.cart import CartWrapper
from aci.envs.graph_coloring import GraphColoring
from aci.components.action_selector import MultiActionSelector
from aci.components.torch_replay_memory import ReplayMemory
from aci.controller.madqn import MADQN
from aci.ploting.training import plot_durations
from aci.ploting.screen import plot_screen
from aci.utils.io import load_yaml
from aci.utils.writer import Writer
import matplotlib
import matplotlib.pyplot as plt
import torch as th


def calc_metrics(rewards, episode_rewards):
    return {
        'rewards': rewards,
        'avg_reward': rewards.mean(), 
        'episode_rewards': episode_rewards,
        'avg_episode_rewards': episode_rewards.mean()
    }


def evaluation(env, controller, writer):
    writer.add_meta(mode='eval', episode_step=0)
    observations = env.reset()
    screens = [env.render()]
    episode_rewards = th.zeros(env.n_agents)

    for t in count():
        writer.add_meta(episode_step=t)

        # Select and perform an action
        actions = controller.get_action(observations)
        observations, rewards, done, _ = env.step(actions)

        screens.append(env.render())
        episode_rewards += rewards

        writer.add_metrics(
            calc_metrics(rewards, episode_rewards),
            {'done': done},
            tf=['avg_reward', 'avg_episode_rewards'] if done else [],
            details_only=~done
        )
        writer.add_frame('eval.observations', lambda: env.render(), details_only=True)
        if done:
            writer.frames_flush()
            break


def train_episode(env, controller, action_selector, memory, writer, batch_size):
    writer.add_meta(mode='train', episode_step=0)
    observations = env.reset()
    episode_rewards = th.zeros(env.n_agents)

    memory.push(observations)
    for t in count():
        writer.add_meta(episode_step=t)

        # Select and perform an action
        proposed_actions = controller.get_q(observations.unsqueeze(0))
        selected_action = action_selector.select_action(proposed_actions)

        observations, rewards, done, _ = env.step(selected_action[0])

        episode_rewards += rewards
        memory.push(observations, selected_action[0], rewards, done)

        writer.add_metrics(
            {**calc_metrics(rewards, episode_rewards), **action_selector.info()},
            {'done': done},
            tf=['avg_reward', 'avg_episode_rewards', 'eps'] if done else [],
            details_only=~done
        )
        writer.add_frame('train.observations', lambda: env.render(), details_only=True)

        if len(memory) > batch_size:
            controller.optimize(*memory.sample(batch_size))

        if done: 
            writer.frames_flush()
            break


def train(
        env, controller, action_selector, memory, writer, 
        num_episodes, target_update, eval_period, batch_size):
    for i_episode in range(num_episodes):
        writer.add_meta(_step=i_episode, episode=i_episode)
        is_eval = i_episode % eval_period == 0
        writer.set_details(is_eval)

        train_episode(env, controller, action_selector, memory, writer, batch_size)

        if i_episode % target_update == 0:
            controller.update()

        if is_eval:
            evaluation(env, controller, writer)

        controller.log(writer)

envs = {
    'cart': CartWrapper,
    'graph_coloring': GraphColoring
}


def main(
        controller_args, env_class, selector_args, train_args, env_args, 
        replay_memory, _output_path, _meta):

    # if gpu is to be used
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    writer = Writer(_output_path, **_meta)

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


if __name__ == "__main__":
    arguments = docopt(__doc__)
    parameter_file = arguments['PARAMETER_FILE']
    output_path = arguments['OUTPUT_PATH']
    parameter = load_yaml(parameter_file)
    main(_output_path=output_path, **parameter)
