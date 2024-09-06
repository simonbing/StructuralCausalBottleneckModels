import numpy as np


def sample_from_simplex(rs, n):
    w = rs.exponential(scale=1.0, size=n)
    return np.expand_dims(w/sum(w), 1)


def sample_convex_comb_bottleneck(rs, d_micro, d_bottleneck):
    w = sample_from_simplex(rs, d_micro)

    print(f'w:\n{w}')

    def f(x):
        return x @ w

    return f
