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


def run_episode(*, env, controller, action_selector, memory=None, writer, test_mode=False):
    observations = env.reset()
    episode_rewards = th.zeros(env.n_agents)

    # added
    if not test_mode:
        controller.init_episode(observations)

    for t in count():
        writer.add_meta(episode_step=t, **action_selector.info())

        # Select and perform an action
        proposed_actions = controller.get_q(observations.unsqueeze(0))
        selected_action = action_selector.select_action(proposed_actions, test_mode=test_mode)

        observations, rewards, done, _ = env.step(selected_action[0])

        episode_rewards += rewards
        if not test_mode:
            controller.update(selected_action[0], observations, rewards, done)

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

    # if len(memory) > batch_size:
    #     controller.optimize(*memory.sample(batch_size))


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
        num_episodes, eval_period):
    for i_episode in range(num_episodes):
        writer.add_meta(_step=i_episode, episode=i_episode)
        is_eval = i_episode % eval_period == 0
        writer.set_details(is_eval)

        train_episode(env, controller, action_selector, memory, writer)

        # if i_episode % target_update == 0:
        #     controller.update()

        if is_eval:
            evaluation(env, controller, action_selector, writer)

        controller.log(writer, is_eval)

def main(
        controller_args, env_class, controller_class, selector_args, train_args, env_args, 
        replay_memory, writer):

    # if gpu is to be used
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    env = ENVIRONMENTS[env_class](**env_args, device=device)
    env.reset()
    n_actions = env.n_actions
    observation_shape = env.observation_shape
    # memory = ReplayMemory(replay_memory)
    memory = None

    controller = CONTROLLERS[controller_class](
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
