from .tabularq import TabularQ
from .heuristic import HeuristicController
from .madqn import MADQN


CONTROLLERS = {
    "madqn": MADQN,
    "heuristic": HeuristicController,
    "tabularq": TabularQ
}
