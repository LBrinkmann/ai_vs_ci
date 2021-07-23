from .neighbor import NeighborView
from .identity import IdentityView
from .encoder import ViewEncoder
from .network import NetworkView

OBSERVER = {
    "neighbor": NeighborView,
    "identity": IdentityView,
    "central": ViewEncoder,
    "network": NetworkView
}
