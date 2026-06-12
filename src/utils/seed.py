"""
src/utils/seed.py
Utilities for reproducible training runs.
"""
import random
import os
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds for python, numpy, and torch for reproducibility.

    Args:
        seed: integer seed
        deterministic: if True, set cudnn.deterministic=True and benchmark=False
    """
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
