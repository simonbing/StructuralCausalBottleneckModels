import numpy as np

from cbm.utils import make_iterable
from cbm.data.utils import sample_mlp


def constant_scalar_mechanism(rs, d_bottleneck, d_micro):
    """
    Sums bottleneck values and multiplies by a scalar. Only use when
    d_bottleneck == 1, otherwise there is a rank issue!

    Args:
        d_bottleneck: int or list[ints]
            Value of bottleneck (i.e. input) dimensions of parent nodes

        d_micro: int
            Dimension of target macro node

    Returns:
        f: function
            Mechanism function
    """
    if d_bottleneck is None:  # No parents
        return None
    else:
        d_bottleneck = make_iterable(d_bottleneck)
        # Sample constant values. Just using 1 for now.
        # constants = np.ones_like(d_bottleneck)
        constants = rs.choice((1, 2, 3, 4), size=len(d_bottleneck))

        def f(*args):
            intermed = []
            for i, arg in enumerate(args):
                intermed.append(arg @ (constants[i] * np.ones((d_bottleneck[i], d_micro))))

            return np.sum(intermed, axis=0)

        return f


def linear_mechanism(rs, d_bottleneck, d_micro):
    if d_bottleneck is None:  # No parents
        return None
    else:
        d_bottleneck = make_iterable(d_bottleneck)

        # Sample one matrix for each incoming bottleneck
        w_list = []
        for i in range(len(d_bottleneck)):
            w = rs.uniform(size=(d_bottleneck[i], d_micro))
            while np.linalg.matrix_rank(w) < d_bottleneck[i]:
                w = rs.uniform(size=(d_bottleneck[i], d_micro))
            w_list.append(w)

        def f(*args):
            intermed = []
            for i, arg in enumerate(args):
                intermed.append(arg @ w_list[i])

            return np.sum(intermed, axis=0)

        return f


def manual_nonlinear_mechanism(rs, d_bottleneck, d_micro):
    # f_lin = linear_mechanism(rs, d_bottleneck, d_micro)
    #
    # def f(*args):
    #     y = f_lin(*args)
    #     # y = y / 1000
    #     return y**3
    # return f

    if d_bottleneck is None:  # No parents
        return None
    else:
        d_bottleneck = make_iterable(d_bottleneck)

        # Sample one matrix for each incoming bottleneck
        w_list = []
        for i in range(len(d_bottleneck)):
            w = rs.uniform(size=(d_bottleneck[i], d_micro))
            while np.linalg.matrix_rank(w) < d_bottleneck[i]:
                w = rs.uniform(size=(d_bottleneck[i], d_micro))
            w_list.append(w)

        def f(*args):
            intermed = []
            for i, arg in enumerate(args):
                intermed.append((arg @ w_list[i]) ** 3)

            return np.sum(intermed, axis=0)

        return f


def sample_nonlin_mechanism(rs, d_bottleneck, d_micro):
    # Debug
    hidden_layers = 4  # 2
    nonlin = 'relu'
    # nonlin = 'leaky_relu'
    # nonlin = 'sigmoid'
    # nonlin = 'swish'
    # nonlin = 'none'
    ###
    if d_bottleneck is None:  # No parents
        return None
    else:
        d_bottleneck = make_iterable(d_bottleneck)

        # Sample one mlp for each incoming bottleneck
        mlp_list = []
        for i in range(len(d_bottleneck)):
            mlp_list.append(sample_mlp(rs=rs, in_dim=d_bottleneck[i],
                                       out_dim=d_micro, hidden_dim=d_bottleneck[i],
                                       hidden_layers=hidden_layers, nonlinearity=nonlin))

        def f(*args):
            intermed = []
            for i, arg in enumerate(args):
                intermed.append(mlp_list[i](arg))

            return np.sum(intermed, axis=0)

        return f
