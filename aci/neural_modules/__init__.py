from .gru import GRUAgent
from .gru_attention import AttentionGRUAgent

NETS = {
    "gru": GRUAgent,
    "attention_gru": AttentionGRUAgent,
}