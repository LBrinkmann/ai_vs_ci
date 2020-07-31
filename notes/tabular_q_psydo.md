# Tabular Q implementation


```
# init controller
Q = 0

# init action selector
eps = ...

# start train

for i in n_episodes:
    # reset environment
    obs = current obs from environment
    # init episode for controller
    prev_obs = obs

    for t in n_steps:
        # get q from controller
        proposed_action = Q[obs]
        # select action (action selector)
        action = proposed_action or random

        # step environment
        reward = step environment with selected_action

        obs = new current obs from environment

        # update controller
        Q[prev_obs, action] = (1 - alpha) * Q[prev_obs, action] + alpha * (reward + gamma * max_over_action(Q[obs])


```

[prev_observations_idx, observations_idx]


[[95030, 286, 95290, 47528, 26, 286], [98501, 69295, 98527, 83885, 69269, 69295]]

[[6480, 332, 6516, 3564, 72, 44], [6540, 5003, 6549, 5811, 4938, 4931]]

[[98501, 69295, 98527, 83885, 69269, 69295], [98368, 91008, 98434, 62110, 58456, 91008]]

[[6540, 5003, 6549, 5811, 4938, 4931], [6533, 6290, 6533, 4373, 4373, 6290]]