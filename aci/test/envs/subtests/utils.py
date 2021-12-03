import torch as th


def random_step(env):
    n_nodes = env.n_nodes
    n_actions = env.n_actions
    ai_color = th.randint(n_actions, size=(n_nodes,))
    ci_color = th.randint(n_actions, size=(n_nodes,))
    rewards, done, info = env.step(
        {'ai': ai_color, 'ci': ci_color}
    )
    return rewards, done, info
