"""Utility functions for Anchors local implementation.

Taken from the original Anchors paper (https://ojs.aaai.org/index.php/AAAI/article/view/11491)."""

import numpy as np


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
