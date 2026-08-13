"""Reproducibility helpers.

XGBoost and scikit-learn also accept an explicit ``random_state`` at
estimator construction; callers should still pass the same seed there.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seeds(seed: int = 42) -> None:
    """Seed Python, NumPy, and hash randomization.

    Parameters
    ----------
    seed:
        Non-negative integer used across libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
