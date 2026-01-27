import numpy as np

from cbm.data.utils import sample_mlp


def sample_from_simplex(rs, n):
    w = rs.exponential(scale=1.0, size=n)
    return np.expand_dims(w/sum(w), 1)


def sample_convex_comb_bottleneck(rs, d_micro, d_bottleneck):
    w = sample_from_simplex(rs, d_micro)

    def f(x):
        return x @ w

    return f


def sample_lin_bottleneck(rs, d_micro, d_bottleneck):
    """
    Samples a (rank d_bottleneck) matrix as the bottleneck function.
    """
    w = rs.uniform(size=(d_micro, d_bottleneck))

    # Make sure rank is correct
    while np.linalg.matrix_rank(w) < d_bottleneck:
        w = rs.uniform(size=(d_micro, d_bottleneck))

    def f(x):
        return x @ w

    return f


def sample_nonlin_bottleneck(rs, d_micro, d_bottleneck):
    hidden_layers = 2  # 2
    # nonlin = 'relu'
    # nonlin = 'leaky_relu'
    # nonlin = 'sigmoid'
    nonlin = 'swish'
    # nonlin = 'none'

    return sample_mlp(rs=rs, in_dim=d_micro, out_dim=d_bottleneck,
                      hidden_dim=d_micro, hidden_layers=hidden_layers,
                      nonlinearity=nonlin)


def manual_nonlinear(rs, d_micro, d_bottleneck):
    """
    Returns a hard-coded nonlinear, surjective function, mainly for debugging.
    """
    w = rs.uniform(size=(d_micro, d_bottleneck))
    # Make sure rank is correct
    while np.linalg.matrix_rank(w) < d_bottleneck:
        w = rs.uniform(size=(d_micro, d_bottleneck))

    def f(x):
        y = x @ w
        
        return y**2

    return f
