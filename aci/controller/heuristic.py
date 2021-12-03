from aci.heuristics import HEURISTICS


class HeuristicController:
    def __init__(self, observation_shape, env_info, device, heuristic_name, agent_args={}, **_):
        self.heuristic = HEURISTICS[heuristic_name](
            observation_shape, **env_info, **agent_args, device=device)
        self.sample_args = None

    def get_q(self, **view):
        return self.heuristic.get_q(**view)

    def init_episode(self, *_, **__):
        pass

    def update(self, *_, **__):
        pass

    def log(self, *_, **__):
        pass
