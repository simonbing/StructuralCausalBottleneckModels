import numpy as np
import wandb

from cbm.data import SCBMSampler
from cbm.estimation import estimate_bottleneck_fcts
from cbm.eval import linear_bottleneck_eval, nonlinear_bottleneck_eval


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
    # Move these params to flags at some point
    SEED = 0
    N_SAMPLES = 50000
    D_MACRO = 3
    D_MICRO = 4
    D_BOTTLENECK = 2
    BOTTLENECK_MODE = 'manual_nonlinear'
    MECH_MODE = 'manual_nonlinear'
    P = 0.99
    ESTIMATION_MODE = 'mlp'

    # wandb stuff
    logging = False

    wandb_config = dict(
        seed=SEED,
        n_samples=N_SAMPLES,
        d_macro=D_MACRO,
        d_micro=D_MICRO,
        d_bottleneck=D_BOTTLENECK,
        bottleneck_mode=BOTTLENECK_MODE,
        mech_mode=MECH_MODE,
        p=P,
        estimator=ESTIMATION_MODE
    )

    wandb.init(
        project='cbm',
        entity='bings',  # this is the team name in wandb
        mode='online' if logging else 'offline',
        config=wandb_config
    )

    sampler = SCBMSampler(seed=SEED,
                          d_macro=D_MACRO,
                          d_micro=D_MICRO,
                          d_bottleneck=D_BOTTLENECK,
                          bottleneck_mode=BOTTLENECK_MODE,
                          mech_mode=MECH_MODE,
                          p=P)

    test_scbm = sampler.sample()

    obs_sample = test_scbm.sample(size=N_SAMPLES)

    # mode = 'linear'
    # mode = 'reduced_rank'
    mode = 'mlp'

    estimated_bottlenecks = estimate_bottleneck_fcts(test_scbm, mode=ESTIMATION_MODE)

    # Apply learned bottlenecks
    estimated_bottleneck_samples = np.empty_like(estimated_bottlenecks, dtype=object)
    for i in range(test_scbm.A.shape[0]):
        for j in range(test_scbm.A.shape[0]):
            if estimated_bottlenecks[i, j] is not None:
                estimated_bottleneck_samples[i, j] = estimated_bottlenecks[i, j](test_scbm.variables[i].value)

    # Evaluation
    if ESTIMATION_MODE == 'mlp':
        eval_mat = nonlinear_bottleneck_eval(estimated_bottleneck_samples,
                                             test_scbm.bottleneck_samples)
    else:
        eval_mat = linear_bottleneck_eval(estimated_bottleneck_samples,
                                          test_scbm.bottleneck_samples)
    mean_score = np.mean(eval_mat[eval_mat != np.array(None)])

    a = 0


if __name__ == '__main__':
    main()