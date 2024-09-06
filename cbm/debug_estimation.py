import numpy as np

from cbm.data import SCBMSampler
from cbm.estimation import estimate_bottleneck_fcts
from cbm.eval import linear_bottleneck_eval


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

    # Apply learned bottlenecks
    estimated_bottleneck_samples = np.empty_like(estimated_bottlenecks, dtype=object)
    for i in range(test_scbm.A.shape[0]):
        for j in range(test_scbm.A.shape[0]):
            if estimated_bottlenecks[i, j] is not None:
                estimated_bottleneck_samples[i, j] = estimated_bottlenecks[i, j](test_scbm.variables[i].value)

    # Evaluation
    eval_mat = linear_bottleneck_eval(estimated_bottleneck_samples, test_scbm.bottleneck_samples)
    mean_score = np.mean(eval_mat[eval_mat != np.array(None)])

    a = 0


if __name__ == '__main__':
    main()