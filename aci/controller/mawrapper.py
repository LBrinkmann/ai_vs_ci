from itertools import permutations
import torch as th
import numpy as np


def shift_obs(tensor_dict):
    """
    Creates previous and current observations.

    Args:
        tensor_dict: each tensor need to have the episode dimension at second position
    """
    previous = {k: v[:, :-1] for k, v in tensor_dict}
    current = {k: v[:, 1:] for k, v in tensor_dict}
    return previous, current


def map_agents(agent_map, *tensors):
    return (th.gather(t, 2, agent_map) for t in tensors)


def map_agents_back(agent_map, *tensors):
    agent_map_inv = th.argsort(agent_map, dim=-1)
    return (th.gather(t, 2, agent_map_inv) for t in tensors)


class MultiAgentWrapper(nn.Module):
    def __init__(
            self, *, weight_sharing, mix_weights_args=None, action_permutation=False, env_info,
            observation_shape):

        super(MultiAgentWrapper, self).__init__()
        self.view_size = view_size
        self.control_size = control_size
        self.weight_sharing = weight_sharing

        self.n_agents = 1 if weight_sharing else n_agents

        self.models = nn.ModuleList([
            NETS[net_type](
                n_inputs=self.n_inputs, input_size=self.input_size, n_actions=n_actions, device=device,
                control_size=self.control_size, **kwargs)
            for n in range(effective_agents)
        ])
        self.multi_type = multi_type
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.device = device
        self.mix_weights_args = mix_weights_args

        # align all weights
        if mix_weights_args is not None:
            ref_model = self.models[0].state_dict()
            for m in self.models:
                m.load_state_dict(ref_model)
        if action_permutation:
            self.permutations = th.tensor(list(permutations(range(n_actions))), device=device)

    def pre_optimise(episode,):
        self.policy_net.reset()
        if (self.mix_freq is not None) and training and (episode % self.mix_freq == 0):
            self.policy_net.mix_weights()
        if training and (episode % self.target_update_freq == 0):
            self._update_target()

    def mix_weights(self):
        return self._mix_weights(**self.mix_weights_args)

    def _mix_weights(self, noise_factor=0.1, mixing_factor=0.8):
        """
        """
        stacked_state_dict = {}
        state_names = list(self.models[0].state_dict().keys())

        stacked_state_dict = {
            k: th.stack(
                tuple(m.state_dict()[k] for m in self.models),
                dim=0
            ) for k in state_names
        }

        stacked_state_dict = {
            k: v + th.randn_like(v) * v.std() * noise_factor
            for k, v in stacked_state_dict.items()
        }

        stacked_state_dict = {
            k: v * (1-mixing_factor) + v.mean(dim=0) * mixing_factor
            for k, v in stacked_state_dict.items()
        }

        for i in range(len(self.models)):
            self.models[i].load_state_dict({k: v[i] for k, v in stacked_state_dict.items()})

    def forward(self, view, mask, agent_map, control=None, control_int=None):
        """
            view: h s p n i
            control: h s p c
            mask: h s p n
            agent_map: h s p
            control_int: h s p
        """
        h, s, p, n, i = view.shape

        if self.share_weights:
            _view, _mask, _control = map_agents(agent_map, view, mask, control)
            q = [
                model(_view[:, :, i], _mask[:, :, i], _control[:, :, i])
                for i, model in enumerate(self.models)
            ]
            q = th.stack(q)
            q = map_agents_back(agent_map, q)
        else:
            q = [
                model(view[:, :, i], mask[:, :, i], control[:, :, i])
                for i in range(p)
            ]
            q = th.stack(q)

        if self.control_type == 'permute':
            assert control_int.max() < len(self.permutations), \
                'Max seed needs to be smaller then the factorial of actions.'
            permutations = self.permutations[control_int]
            q = th.gather(q, -1, permutations)

        return q

    def log(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        for m in self.models:
            m.reset()
