import gym
import torch as th
import numpy as np
import torchvision.transforms as T
from PIL import Image


resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_cart_location(env, screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen(env, device):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(env, screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = th.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)


class CartWrapper():
    def __init__(self, device, n_agents, max_steps):
        self.envs = [gym.make('CartPole-v0').unwrapped for i in range(n_agents)]
        self.device = device
        self.n_agents = n_agents

    def step(self, actions):
        rewards = th.zeros(self.n_agents)
        dones = th.zeros(self.n_agents, dtype=th.bool)
        for i, env in enumerate(self.envs):
            _, reward, done, _ = env.step(actions[i].item())
            rewards[i] = reward
            dones[i] = done
        new_screen = th.cat([get_screen(env, self.device) for env in self.envs])
        observation = self.last_screen - new_screen
        self.last_screen = new_screen
        return observation, rewards, dones.any(), {}
    
    def render(self):
        frame = th.stack([get_screen(env, self.device) for env in self.envs])
        return frame

    def reset(self):
        for env in self.envs:
            env.reset()
        self.last_screen = th.cat([get_screen(env, self.device) for env in self.envs])
        return th.zeros_like(self.last_screen)

    @property
    def n_actions(self):
        return self.envs[0].action_space.n

    @property
    def observation_shape(self):
        return self.last_screen.shape[1:]

    def __del__(self):
        for env in self.envs:
            env.close()  