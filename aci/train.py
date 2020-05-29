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
from aci.envs.cart import CartWrapper
from aci.controller.dqn import DQN, ActionSelector
from aci.ploting.training import plot_durations
from aci.ploting.screen import plot_screen
from aci.utils.io import load_yaml
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter


def divobservation(env):
    last_screen = env.observe()
    last_state = last_screen - last_screen
    while True:
        current_screen = env.observe()
        try:
            current_state = current_screen - last_screen
        except: 
            current_state = None
        yield last_state, current_state, current_screen
        last_screen = current_screen
        last_state = current_state


def evaluation(env, controller):
    env.reset()
    last_state, current_state, screen = next(divobservation(env))
    screens = []
    episode_reward = 0
    for t in count():
        # Select and perform an action
        action = controller.get_action(current_state)
        reward, done = env.step(action)

        last_state, current_state, screen = next(divobservation(env))

        screens.append(screen)
        episode_reward += reward[0]
        if done:
            break
    video = torch.cat(screens).unsqueeze(0)
    return {'episode_reward': episode_reward, 'episode_steps': t+1}, video


def add_metrics(d, scope, **kwargs):
    return {k: {**d.get(k, {}), scope: v} for k, v in kwargs.items()}

def write_metrics(metrics_dict, writer, i_episode):
    for k, v in metrics_dict.items():
        writer.add_scalars(k, v, i_episode)


def train(env, controller, action_selector, writer, num_episodes, target_update, eval_period):
    total_steps = 0
    episode_durations = []
    for i_episode in range(num_episodes):
        episode_reward = 0

        # Initialize the environment and state
        env.reset()

        last_screen = env.observe()
        current_screen = env.observe()
        state = current_screen - last_screen

        # last_state, current_state, _ = next(divobservation(env))
        for t in count():
            # Select and perform an action
            proposed_action = controller.get_action(state)
            selected_action = action_selector.select_action(proposed_action)
            reward, done = env.step(selected_action)

            # Observe new state
            last_screen = current_screen
            current_screen = env.observe()
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None


            # last_state, current_state, _ = next(divobservation(env))

            controller.push_transition(state, selected_action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            controller.optimize()

            total_steps += 1
            episode_reward += reward[0]

            if done:
                episode_durations.append(t + 1)
                plot_durations(episode_durations)
                break
        
        # metrics_dict = add_metrics(
        #     {}, 'train', episode_reward=episode_reward, episode_steps=t+1)

        # Update the target network, copying all weights and biases in DQN
        if i_episode % target_update == 0:
            controller.update()

        # # eval and logging
        # if i_episode % eval_period == 0:
        #     eval_metrics_dict, video  = evaluation(env, controller)
        #     writer.add_video('eval_play', video, i_episode)
        
        # metrics_dict = add_metrics(
        #     metrics_dict, 'train', **eval_metrics_dict)
        
        # write_metrics(metrics_dict, writer, i_episode)
        # controller.log(writer, i_episode, i_episode % eval_period == 0)
        # action_selector.log(writer, i_episode, i_episode % eval_period == 0)


def main(opt_args, selector_args, train_args, replay_memory):
    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter()
    env = CartWrapper(device)
    env.reset()
    n_actions = env.n_actions
    observation_shape = env.observation_shape

    controller = DQN(
        observation_shape=observation_shape,
        n_actions=n_actions,
        opt_args=opt_args, 
        replay_memory=replay_memory,
        device=device)

    action_selector = ActionSelector(device=device, **selector_args)

    train(env, controller, action_selector, writer, **train_args)

    print('Complete')
    env.env.render()
    env.env.close()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    arguments = docopt(__doc__)
    parameter = load_yaml(arguments['PARAMETER_YAML'])
    main(**parameter)
