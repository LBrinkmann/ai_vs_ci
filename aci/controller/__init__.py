CONTROLLERS = {}

# from .dqn import DQN
from .madqn import MADQN
from .heuristic import HeuristicController
from .tabularq import TabularQ


CONTROLLERS = {
    # "dqn": DQN,
    "madqn": MADQN,
    "heuristic": HeuristicController,
    "tabularq": TabularQ
}