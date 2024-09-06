import numpy as np

from cbm.utils import make_iterable


def constant_scalar_mechanism(d_bottleneck, d_micro):
    """
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
        d_bottleneck = make_iterable(d_bottleneck)  # this should be all ones
        # Sample constant values. Just using 1 for now.
        constants = np.ones_like(d_bottleneck)

        def f(*args):
            intermed = []
            for i, arg in enumerate(args):
                intermed.append(arg @ (constants[i] * np.ones((1, d_micro))))

            return np.sum(intermed, axis=0)

        return f
