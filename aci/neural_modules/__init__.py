from .gru_pooling import PoolingGRUAgent
from .gcn import GCNModel

NETS = {
    "pooling_gru": PoolingGRUAgent,
    "gcn": GCNModel,
}
