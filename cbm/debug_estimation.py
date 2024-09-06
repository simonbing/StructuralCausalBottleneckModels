import numpy as np

from cbm.data import SCBMSampler
from cbm.estimation import estimate_bottleneck_fcts


def check_open_path(source, target):
    """
    Check if there is an open path between the start and
    end nodes. Probably this is pretty inefficient.
    """
    if source in target.parents:
        return True

    # Check that candidate node has parents
    if target.parents is not None:
        # Only consider candidates that themselves have parents
        candidates = [i for i in target.parents if i.parents is not None]
        for candidate in candidates:
            intermed = check_open_path(source, candidate)
            if intermed:
                return True
    else:
        return False

    return False


def main():
    sampler = SCBMSampler(seed=0,
                          d_macro=4,
                          d_micro=4,
                          d_bottleneck=1,
                          bottleneck_mode='convex_comb',
                          mech_mode='constant',
                          p=0.8)

    test_scbm = sampler.sample()

    obs_sample = test_scbm.sample(size=50000)

    estimated_bottlenecks = estimate_bottleneck_fcts(test_scbm)

    a = 0


if __name__ == '__main__':
    main()