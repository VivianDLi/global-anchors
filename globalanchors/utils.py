"""General utility functions for all modules."""

import numpy as np


def normalize_feature_indices(x):
    return tuple(sorted(x))


def exp_normalize(x: np.array) -> np.array:
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()
