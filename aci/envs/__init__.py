ENVIRONMENTS = {}

from .cart import CartWrapper
from .graph_coloring import GraphColoring, GraphColoringPartiallyFixed
from .adversial_graph_coloring import AdGraphColoring
from .mcar import CarWrapper

ENVIRONMENTS = {
    'cart': CartWrapper,
    'graph_coloring': GraphColoring,
    'partially_graph_coloring': GraphColoringPartiallyFixed,
    'adversial_graph_coloring': AdGraphColoring,
    'mountain_car': CarWrapper
}

