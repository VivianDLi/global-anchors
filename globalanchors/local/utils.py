"""Utility functions for Anchors local implementation.

Taken from the original Anchors paper (https://ojs.aaai.org/index.php/AAAI/article/view/11491)."""

import numpy as np
from numpy.linalg import norm

from globalanchors.types import DistanceFunctionType


def bernoulli_kl(p: float, q: float) -> float:
    p = min(0.9999999999999999, max(0.0000001, p))
    q = min(0.9999999999999999, max(0.0000001, q))
    return p * np.log(float(p) / q) + (1 - p) * np.log(float(1 - p) / (1 - q))


def bernoulli_lb(p: float, level: float) -> float:
    lm = p
    um = min(min(1, p + np.sqrt(level / 2.0)), 1)
    qm = (um + lm) / 2.0
    #         print 'lm', lm, 'qm', qm, kl_bernoulli(p, qm)
    if bernoulli_kl(p, qm) > level:
        um = qm
    else:
        lm = qm
    return um


def bernoulli_ub(p: float, level: float) -> float:
    um = p
    lm = max(min(1, p - np.sqrt(level / 2.0)), 0)
    qm = (um + lm) / 2.0
    #         print 'lm', lm, 'qm', qm, kl_bernoulli(p, qm)
    if bernoulli_kl(p, qm) > level:
        lm = qm
    else:
        um = qm
    return lm


def compute_beta(n_features: int, t: int, delta: float) -> float:
    alpha = 1.1
    k = 405.5
    temp = np.log(k * n_features * (t**alpha) / delta)
    return temp + np.log(temp)


def exp_normalize(x: np.array) -> np.array:
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()


def normalized_cosine_distance(x1: np.array, x2: np.array) -> float:
    return (
        1 - (np.dot(x1, x2) / (norm(x1) * norm(x2)))
    ) / 2  # range of [0, 1]


def normalized_squared_euclidean_distance(x1: np.array, x2: np.array) -> float:
    """Taken from https://reference.wolfram.com/language/ref/NormalizedSquaredEuclideanDistance.html."""
    x1_p = x1 - np.mean(x1)
    x2_p = x2 - np.mean(x2)
    return (
        0.5 * norm(x1_p - x2_p) ** 2 / (norm(x1_p) ** 2 + norm(x2_p) ** 2)
    )  # range of [0, 1]


def get_distance_function(distance_type: DistanceFunctionType):
    match distance_type:
        case "cosine":
            return normalized_cosine_distance
        case "neuclid":
            return normalized_squared_euclidean_distance
        case _:
            raise ValueError(
                f"Distance function {distance_type} not recognized."
            )
