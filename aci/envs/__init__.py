from .table_game import TableGame
from .network_game import NetworkGame


ENVIRONMENTS = {
    'network_game': NetworkGame,
    'table_game': TableGame
}
