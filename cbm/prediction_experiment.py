"""
Runs an experiment to predict the values of macro nodes (effect estimation).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import wandb

from cbm.data import SCBMSampler
from cbm.estimation import estimate_effects_ols, estimate_bottleneck_and_mechanism_fcts
from cbm.estimation.utils import _get_var_idx
from cbm.SCBM_models import get_SCBM_tf_1


def single_prediction_run(seed, n_samples, d_macro, d_micro, d_bottleneck,
                          bottleneck_mode, mech_mode, p, estimation_mode, predictors):
    # Move these params to flags at some point
    PREDICTOR = predictors

    # wandb stuff
    logging = False

    wandb_config = dict(
        seed=seed,
        n_samples=n_samples,
        d_macro=d_macro,
        d_micro=d_micro,
        d_bottleneck=d_bottleneck,
        bottleneck_mode=bottleneck_mode,
        mech_mode=mech_mode,
        p=p,
        estimator=estimation_mode,
        predictors=predictors
    )

    wandb.init(
        project='cbm',
        entity='bings',  # this is the team name in wandb
        mode='online' if logging else 'offline',
        config=wandb_config
    )

    # sampler = SCBMSampler(seed=seed,
    #                       d_macro=d_macro,
    #                       d_micro=d_micro,
    #                       d_bottleneck=d_bottleneck,
    #                       bottleneck_mode=bottleneck_mode,
    #                       mech_mode=mech_mode,
    #                       p=p)
    #
    # test_scbm = sampler.sample()

    ### Hardcoded SCBM goes here
    test_scbm = get_SCBM_tf_1(seed=seed, d=d_micro)

    # n_train = int(0.8 * n_samples)
    # n_test = int(0.2 * n_samples)
    n_train = n_samples
    n_test = 1000

    # Train sample last since these are saved in SCBM object and used for estimation
    test_sample, test_bn_sample = test_scbm.sample(size=n_test)
    train_sample, train_bn_sample = test_scbm.sample(size=n_train)

    out_dict = {}

    for predictor in predictors:
        # TODO: move this its own function
        # Fit the chosen estimator(s)
        if predictor == 'ols':
            estimated_effect_fcts = estimate_effects_ols(test_scbm, test_sample)
        elif predictor == 'bottleneck':
            estimated_bottleneck_fcts, estimated_mechanism_fcts = estimate_bottleneck_and_mechanism_fcts(
                test_scbm, mode='linear')

            dim = test_scbm.A.shape[0]
            estimated_effect_fcts = np.empty_like(estimated_bottleneck_fcts)

            # Function factory
            def make_effect_fct(i, j):
                bottleneck_fct = estimated_bottleneck_fcts[i, j]
                mechanism_fct = estimated_mechanism_fcts[i, j]

                def effect_fct(x):
                    return mechanism_fct(bottleneck_fct(x))
                return effect_fct

            for i in range(dim):
                for j in range(dim):
                    if estimated_bottleneck_fcts[i, j] is not None:
                        estimated_effect_fcts[i, j] = make_effect_fct(i, j)

        # Get errors for all predictions
        estimates = np.empty(test_scbm.A.shape[0], dtype=object)

        # Loop over variables (in causal ordering) and get predictions from parents
        causal_order = np.arange(test_scbm.A.shape[0])

        for target_idx in causal_order:
            target = test_scbm.variables[target_idx]

            if target.parents is None:
                pass
            else:
                parent_idxs = [_get_var_idx(test_scbm, parent) for parent in
                               target.parents]

                # Estimates from individual parents are summed
                estimate = sum([estimated_effect_fcts[parent_idx, target_idx](
                    test_sample[parent_idx])
                                for parent_idx in parent_idxs])
                estimates[target_idx] = estimate

        # Compute metrics
        metrics_list = np.empty_like(estimates)
        for i, (target, estimate) in enumerate(zip(test_sample, estimates)):
            if estimate is not None:  # this skips root nodes
                mse = mean_squared_error(target, estimate)

                mae = mean_absolute_error(target, estimate)

                error = target - estimate
                cov = np.cov(error.T)

                metrics = {'mse': mse,
                           'mae': mae,
                           'cov': cov}
                metrics_list[i] = metrics

        out_dict[predictor] = metrics_list

    return out_dict


def main():
    # Move these params to flags at some point
    GLOBAL_SEED = 42
    D_MACRO = 3
    D_MICRO = 90
    D_BOTTLENECK = 3
    BOTTLENECK_MODE = 'linear'
    MECH_MODE = 'linear'
    P = 0.99
    ESTIMATION_MODE = 'linear'  # This is the bottleneck predictor
    PREDICTORS = ['bottleneck', 'ols']

    rs = np.random.RandomState(GLOBAL_SEED)
    seeds = rs.randint(low=0, high=1e5, size=5)

    mse_list_bottleneck = []
    mse_list_ols = []
    mae_list_bottleneck = []
    mae_list_ols = []
    cov_list_bottleneck = []
    cov_list_ols = []

    PRED_NODE = 2  # which node to plot

    for seed in seeds:
        # sample_sizes = [100, 200, 1000, 10000, 50000, 100000]
        sample_sizes = [100, 200, 300, 500, 1000]

        output_list = []
        for sample_size in sample_sizes:
            output = single_prediction_run(seed=seed,
                                            n_samples=sample_size,
                                            d_macro=D_MACRO,
                                            d_micro=D_MICRO,
                                            d_bottleneck=D_BOTTLENECK,
                                            bottleneck_mode=BOTTLENECK_MODE,
                                            mech_mode=MECH_MODE,
                                            p=P,
                                            estimation_mode=ESTIMATION_MODE,
                                            predictors=PREDICTORS)
            output_list.append(output)

        mse_list_bottleneck.append([output['bottleneck'][PRED_NODE]['mse'] for output in output_list])
        mse_list_ols.append([output['ols'][PRED_NODE]['mse'] for output in output_list])
        mae_list_bottleneck.append([output['bottleneck'][PRED_NODE]['mae'] for output in output_list])
        mae_list_ols.append([output['ols'][PRED_NODE]['mae'] for output in output_list])
        cov_list_bottleneck.append([np.max(abs(output['bottleneck'][PRED_NODE]['cov'])) for output in output_list])
        cov_list_ols.append([np.max(abs(output['ols'][PRED_NODE]['cov'])) for output in output_list])

    model_data = np.repeat(PREDICTORS, len(sample_sizes) * len(seeds))
    sample_size_data = np.tile(np.repeat([str(x) for x in sample_sizes], len(seeds)), len(PREDICTORS))
    mse_data = np.concatenate((np.asarray(mse_list_bottleneck).flatten(order='F'),
                               np.asarray(mse_list_ols).flatten(order='F')))
    mae_data = np.concatenate((np.asarray(mae_list_bottleneck).flatten(order='F'),
                               np.asarray(mae_list_ols).flatten(order='F')))
    cov_data = np.concatenate((np.asarray(cov_list_bottleneck).flatten(order='F'),
                               np.asarray(cov_list_ols).flatten(order='F')))

    plot_data = pd.DataFrame({'model': model_data,
                              'sample size': sample_size_data,
                              'mse': mse_data,
                              'mae': mae_data,
                              'max(cov)': cov_data})

    x_plot = [1, 2, 3, 4, 5, 6]

    fig = plt.figure()

    ax = sns.lineplot(data=plot_data, x='sample size', y='max(cov)', hue='model')
    plt.title(f'd_x: {D_MICRO}, d_z: {D_BOTTLENECK}, d_macro: {D_MACRO}, '
              f'node: {PRED_NODE+1}, {ESTIMATION_MODE}')

    fig.show()

    a=0


if __name__ == '__main__':
    main()
