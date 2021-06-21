from .neighbor import NeighborView, ViewEncoder
from .identity import IdentityView

OBSERVER = {
    "neighbor": NeighborView,
    "identity": IdentityView,
    "central": ViewEncoder
}
