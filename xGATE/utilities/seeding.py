import random

import numpy as np
import torch


def set_seed(seed):
    """Seed the Python, NumPy and PyTorch global RNGs for reproducible runs.

    xGATE draws from three independent RNG streams: the stdlib ``random`` module
    (subgraph sampling), NumPy (``np.random`` in the random walks) and PyTorch
    (VAE weight initialization and the reparameterization noise). Seeding all
    three from one place is what makes a run reproducible.

    Call this once at the start of a run. Pass ``seed=None`` to leave the RNGs
    untouched -- this is what the inner helpers do so that a single top-level
    seed advances one continuous stream instead of being reset mid-run.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
