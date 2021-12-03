from itertools import permutations
import torch as th
from aci.utils.tensor_op import map_tensors, map_tensors_back
from aci.neural_modules import NETS


def apply_permutations(tensor, control_int, _permutations):
    assert control_int.max() < len(_permutations), \
        'Max seed needs to be smaller then the factorial of actions.'
    _permutations = _permutations[control_int]
    tensor = th.gather(tensor, -1, _permutations)
    return tensor


def create_permutations(n_actions, device):
    return th.tensor(list(permutations(range(n_actions))), device=device)


def safe_select(tensor, idx):
    if tensor is not None:
        return tensor[:, :, idx]
    else:
        return None


def safe_stack(tensor):
    """
    Args:
        tensor: [b s p ...]
    Returns:
        tensor: [b*p s ...]
    """
    if tensor is not None:
        _tensor = tensor.transpose(2, 1)  # b p e ...
        return _tensor.reshape(-1, *_tensor.shape[2:])  # b*p e ...
    else:
        return None


def unstack(tensor, batch_size, n_nodes):
    """
    Args:
        tensor: [b*p e ...]
    Returns:
        tensor: [b e p ...]
    """
    assert tensor.shape[0] == batch_size * n_nodes
    _tensor = tensor.view(batch_size, n_nodes, *tensor.shape[1:])  # b p e ...
    return _tensor.transpose(2, 1)  # b e p ..


class MultiAgentWrapper(th.nn.Module):
    def __init__(
            self, *, weight_sharing, mix_weights_args=None, action_permutation=False, env_info,
            observation_shape, net_type, device, model_args):
        super(MultiAgentWrapper, self).__init__()
        self.weight_sharing = weight_sharing
        self.mix_weights_args = mix_weights_args

        n_neighbors = env_info['max_neighbors']
        n_actions = env_info['n_actions']
        n_agents = env_info['n_nodes']

        view_size = observation_shape['view'][-1]
        control_size = observation_shape['control'][-1]

        effective_agents = 1 if self.weight_sharing else n_agents
        self.models = th.nn.ModuleList([
            NETS[net_type](
                n_neighbors=n_neighbors, n_actions=n_actions, view_size=view_size,
                control_size=control_size, device=device, **model_args)
            for n in range(effective_agents)
        ])

        # align all weights
        if mix_weights_args is not None:
            ref_model = self.models[0].state_dict()
            for m in self.models:
                m.load_state_dict(ref_model)
        if action_permutation:
            self.permutations = create_permutations(n_actions, device)
        else:
            self.permutations = None

    def mix_weights(self):
        if self.mix_weights_args is not None:
            return self._mix_weights(**self.mix_weights_args)

    def _mix_weights(self, noise_factor, mixing_factor):
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
            view: b s p n i
            control: b s p c
            mask: b s p n
            agent_map: b s p
            control_int: b s p
        """
        b, s, p, n, i = view.shape

        if not self.weight_sharing:
            _view, _mask, _control = map_tensors(agent_map, view, mask, control)
            q = [
                model(safe_select(_view, i), safe_select(_mask, i), safe_select(_control, i))
                for i, model in enumerate(self.models)
            ]
            q = th.stack(q, dim=2)
            assert (b, s, p) == q.shape[:-1]
            q, = map_tensors_back(agent_map, q)
        else:
            q = self.models[0](safe_stack(view), safe_stack(mask), safe_stack(control))
            q = unstack(q, b, p)
            assert (b, s, p) == q.shape[:-1]

        if self.permutations is not None:
            q = apply_permutations(q, control_int,  self.permutations)

        return q

    def log(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        for m in self.models:
            m.reset()
