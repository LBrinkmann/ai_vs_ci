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


def run_episode(*, env, controller, action_selector, memory=None, writer, test_mode=False):
    observations = env.reset()
    episode_rewards = th.zeros(env.n_agents)

    if memory is not None:
        memory.next_seq(observations)
    controller.init_episode(observations)
    for t in count():
        writer.add_meta(episode_step=t, **action_selector.info())

        # Select and perform an action
        proposed_actions = controller.get_q(observations.unsqueeze(0))
        selected_action = action_selector.select_action(proposed_actions, test_mode=test_mode)

        observations, rewards, done, _ = env.step(selected_action[0])

        episode_rewards += rewards
        if memory is not None:
            memory.push(observations, selected_action[0], rewards, done)

        writer.add_metrics(
            calc_metrics(rewards, episode_rewards),
            {'done': done},
            tf=['avg_reward', 'avg_episode_rewards'] if done else [],
            details_only=not done
        )
        writer.add_frame('{mode}.observations', lambda: env.render(), details_only=True)
        if done: 
            break
    
    writer.frames_flush()


def train_episode(env, controller, action_selector, memory, writer):
    writer.add_meta(mode='train')
    run_episode(
        env=env, 
        controller=controller, 
        action_selector=action_selector, 
        memory=memory, 
        writer=writer)

    if memory.can_sample():
        controller.optimize(*memory.sample())


def evaluation(env, controller, action_selector, writer):
    writer.add_meta(mode='eval')
    run_episode(
        env=env, 
        controller=controller,
        action_selector=action_selector,
        writer=writer,
        test_mode=True)


def train(
        env, controller, action_selector, memory, writer, 
        num_episodes, target_update, eval_period):
    for i_episode in range(num_episodes):
        writer.add_meta(_step=i_episode, episode=i_episode)
        is_eval = i_episode % eval_period == 0
        writer.set_details(is_eval)

        train_episode(env, controller, action_selector, memory, writer)

        if i_episode % target_update == 0:
            controller.update()

        if is_eval:
            evaluation(env, controller, action_selector, writer)

        controller.log(writer)

envs = {
    'cart': CartWrapper,
    'graph_coloring': GraphColoring
}


def main(
        controller_args, env_class, selector_args, train_args, env_args, 
        memory_args, writer):

    # if gpu is to be used
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    env = envs[env_class](**env_args, device=device)
    env.reset()
    n_actions = env.n_actions
    observation_shape = env.observation_shape
    n_agents = env.n_agents
    memory = ReplayMemory(
        **memory_args, observation_shape=observation_shape, n_agents=n_agents)

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

    meta = {f'label.{k}': v for k, v in parameter.get('labels', {}).items()}
    writer = Writer(output_path, **meta)
    main(writer=writer, **parameter['params'])
