from .gru_pooling import PoolingGRUAgent
from .gcn import GCNModel
from .gru import GRUAgent

NETS = {
    "pooling_gru": PoolingGRUAgent,
    "gcn": GCNModel,
    'gru': GRUAgent
}
