import numpy as np

from cbm.data.utils import sample_mlp


def sample_from_simplex(rs, n):
    w = rs.exponential(scale=1.0, size=n)
    return np.expand_dims(w/sum(w), 1)


def sample_convex_comb_bottleneck(rs, d_micro, d_bottleneck):
    w = sample_from_simplex(rs, d_micro)

    # print(f'w:\n{w}')

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

    # w0 = rs.uniform(size=(d_micro, d_bottleneck))
    # # Make sure rank is correct
    # while np.linalg.matrix_rank(w0) < d_bottleneck:
    #     w0 = rs.uniform(size=(d_micro, d_bottleneck))

    # # See if QR decomposition leads to better behaved function
    # # Q, R = np.linalg.qr(w0)
    # # w0 = Q

    # # w0 = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]]).T

    # w1 = rs.uniform(size=(d_micro, d_micro))
    # # Make sure rank is correct
    # while np.linalg.matrix_rank(w1) < d_micro:
    #     w1 = rs.uniform(size=(d_micro, d_micro))

    # w2 = rs.uniform(size=(d_micro, d_micro))
    # # Make sure rank is correct
    # while np.linalg.matrix_rank(w2) < d_micro:
    #     w2 = rs.uniform(size=(d_micro, d_micro))

    # w3 = rs.uniform(size=(d_micro, d_micro))
    # # Make sure rank is correct
    # while np.linalg.matrix_rank(w3) < d_micro:
    #     w3 = rs.uniform(size=(d_micro, d_micro))

    # w4 = rs.uniform(size=(d_micro, d_micro))
    # # Make sure rank is correct
    # while np.linalg.matrix_rank(w4) < d_micro:
    #     w4 = rs.uniform(size=(d_micro, d_micro))

    # # Q, R = np.linalg.qr(w1)
    # # w1 = Q

    # # Q, R = np.linalg.qr(w2)
    # # w2 = Q

    # # Q, R = np.linalg.qr(w3)
    # # w3 = Q

    # # Q, R = np.linalg.qr(w4)
    # # w4 = Q

    # w = w1 @ w2 @ w3 @ w4 @ w0

    # def leaky_relu(x):
    #     if x < 0:
    #         return 0.1 * x
    #     else:
    #         return x

    # def sigmoid(x):
    #     return 1 / (1 + np.exp(-x))

    # def f(x):
    #     # return np.vectorize(leaky_relu)(x @ w)
    #     return x @ w

    # return f


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
        # y = y / 10
        return y**2

    return f
