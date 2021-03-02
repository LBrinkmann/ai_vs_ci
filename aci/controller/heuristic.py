from aci.heuristics import HEURISTICS


class HeuristicController:
    def __init__(self, observation_shape, env_info, device, heuristic_name, agent_args):
        self.heuristic = HEURISTICS[heuristic_name](
            observation_shape, **env_info, **agent_args, device=device)

    def get_q(self, observation, training=None):
        return self.heuristic.get_q(observation)

    def init_episode(self, *_, **__):
        pass

    def update(self, *_, **__):
        pass

    def log(self, *_, **__):
        pass
