CONTROLLERS = {}

# from .dqn import DQN
from .madqn import MADQN
from .maq import MAQ
from .heuristic import HeuristicController
from .tabularq import TabularQ


CONTROLLERS = {
    "maq": MAQ,
    "madqn": MADQN,
    "heuristic": HeuristicController,
    "tabularq": TabularQ
}