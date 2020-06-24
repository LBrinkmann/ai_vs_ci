CONTROLLERS = {}

# from .dqn import DQN
from .madqn import MADQN
from .simple_graph import SimpleGraphAgent
from .tabularq import TabularQ


CONTROLLERS = {
    # "dqn": DQN,
    "madqn": MADQN,
    "sgraph": SimpleGraphAgent,
    "tabularq": TabularQ
}