from .gru_pooling import PoolingGRUAgent
from .gcn import GCNModel
from .gru import GRUAgent
from .central import CentralAgent

NETS = {
    "pooling_gru": PoolingGRUAgent,
    "gcn": GCNModel,
    'gru': GRUAgent,
    'central': CentralAgent
}
