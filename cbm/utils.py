from collections.abc import Iterable

import numpy as np


def make_iterable(x):
    if isinstance(x, Iterable):
        return x
    else:
        return [x]


def get_cond_submats(target, E):
    """
    Get the submatrices of the covariance for the Gaussian conditioning formula.

    Args:
        target: list or int
            indices of intervention targets
        E: np.array
            original covariance matrix
    Returns:
        E_11, E_22, E_12: np.array
            submatrices of E
    """
    E_11 = np.delete(np.delete(E, target, axis=0), target, axis=1)

    target = make_iterable(target)

    E_22 = E[target, :][:, target]

    E_12 = np.delete(E, target, axis=0)[:, target]

    return E_11, E_22, E_12


def get_cond_mean_cov(target, value, E):
    """
    Calculate the mean and covariance for the Gaussian conditioning formula.

    Args:
        target: list[int] or int
            indices of interventions targets
        value: list
            list of intervention values targets take
        E: np.array
            original covariance matrix
    """
    value = make_iterable(value)

    E_11, E_22, E_12 = get_cond_submats(target, E)

    if isinstance(E_22, np.ndarray):
        E_22_inv = np.linalg.inv(E_22)
    else: # if this is a scalar
        E_22_inv = [1/E_22]

    # Treat distr. as zero mean Gaussian, original mean is added back later
    mu_bar = E_12 @ E_22_inv @ value

    E_bar = E_11 - E_12 @ E_22_inv @ E_12.T

    return mu_bar, E_bar
