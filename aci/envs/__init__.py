ENVIRONMENTS = {}

from .cart import CartWrapper
from .graph_coloring import GraphColoring, GraphColoringPartiallyFixed
from .adversial_graph_coloring import AdGraphColoring

ENVIRONMENTS = {
    'cart': CartWrapper,
    'graph_coloring': GraphColoring,
    'partially_graph_coloring': GraphColoringPartiallyFixed,
    'adversial_graph_coloring': AdGraphColoring
}

