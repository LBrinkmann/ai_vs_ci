from .tabularq import TabularQ
from .heuristic import HeuristicController
from .madqn import MADQN
from .dqn import DQN


CONTROLLERS = {
    "madqn": MADQN,
    "heuristic": HeuristicController,
    "tabularq": TabularQ,
    "dqn": DQN
}
