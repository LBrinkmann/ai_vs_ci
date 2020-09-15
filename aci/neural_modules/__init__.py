from .gru import GRUAgent
from .gru_attention import AttentionGRUAgent
from .gru_pooling import PoolingGRUAgent
from .gcn import GCNModel

NETS = {
    "gru": GRUAgent,
    "attention_gru": AttentionGRUAgent,
    "pooling_gru": PoolingGRUAgent,
    "gcn": GCNModel
}