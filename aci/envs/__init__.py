ENVIRONMENTS = {}

from .cart import CartWrapper
from .graph_coloring import GraphColoring, GraphColoringPartiallyFixed

ENVIRONMENTS = {
    'cart': CartWrapper,
    'graph_coloring': GraphColoring,
    'partially_graph_coloring': GraphColoringPartiallyFixed
}

