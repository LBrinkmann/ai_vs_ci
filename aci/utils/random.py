import torch as th
import numpy as np
import random


def set_seeds(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        th.set_deterministic(True)
