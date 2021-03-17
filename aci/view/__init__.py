from .neighbor import NeighborView
from .neighbor_id import NeighborIDView
from .identity import IdentityView

VIEW = {
    "neighbor": NeighborView,
    "identity": IdentityView,
    "neighbor_id": NeighborIDView
}
