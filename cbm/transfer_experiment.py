import sys
import os

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

from cbm.data import SCBMSampler
from cbm.estimation import estimate_bottleneck_and_mechanism_fcts
from cbm.eval.mlp_regressor import MLPRegressor
from cbm.SCBM_models import get_SCBM_tf_1
from cbm.utils import make_iterable
from cbm.plotting import plot_multiple_transfer_runs

import wandb


def single_transfer_run(seed, SCBM, n_bn_train, n_train, n_test, target_idx,
                        source_idx, cond_idxs, cond_type=['x', 'bn'],
                        mode='linear'):
    cond_idxs = make_iterable(cond_idxs)

    train_samples, train_bn_samples = SCBM.sample(size=n_train)
    test_samples, test_bn_samples = SCBM.sample(size=n_test)

    output_dict = {}

    if 'x' in cond_type:
        # Fit OLS without applying bottlenecks
        if mode == 'linear':
            regr_x = LinearRegression()
        elif mode == 'nonlinear':
            regr_x = MLPRegressor(seed=seed,
                                  d=2*train_samples[0].shape[-1],
                                  d_out=train_samples[0].shape[-1],
                                  dense_layers=[128, 128, 128, 128, 128, 128],
                                  learning_rate=0.0005,
                                  momentum=0.9,
                                  epochs=100,
                                  batch_size=128,
                                  source=None,
                                  target=None
                                  )

        X_cond_x_train = [train_samples[idx] for idx in cond_idxs]
        y_x_train = train_samples[target_idx]

        regr_x.fit(X=np.concatenate([train_samples[source_idx],
                                       *X_cond_x_train], axis=1),
                     y=y_x_train)

        # Test on held out data
        X_cond_x_test = [test_samples[idx] for idx in cond_idxs]
        y_hat_x_test = regr_x.predict(X=np.concatenate([test_samples[source_idx],
                                                            *X_cond_x_test], axis=1))

        metrics_ols = {
            'mse': mean_squared_error(test_samples[target_idx], y_hat_x_test),
            'mae': mean_absolute_error(test_samples[target_idx], y_hat_x_test),
            'var': np.cov(test_samples[target_idx].T - y_hat_x_test.T)
        }

        output_dict['x'] = metrics_ols

    if 'bn' in cond_type:
        # Fit OLS with bottlenecks applied to conditioning set

        # Estimate bottleneck functions
        bn_train_samples, bn_train_bn_samples = SCBM.sample(size=n_bn_train)
        bn_train_samples = train_samples

        bn_mode = 'linear' if mode == 'linear' else 'mlp'
        estimated_bn_fcts, estimated_mech_fcts = \
            estimate_bottleneck_and_mechanism_fcts(SCBM, bn_train_samples, mode=bn_mode)
        if mode == 'linear':
            regr_bn = LinearRegression()
        elif mode == 'nonlinear':
            regr_bn = MLPRegressor(seed=seed,
                                   d=502,  # TODO: automatically calculate this!
                                   d_out=train_samples[0].shape[-1],
                                   dense_layers=[128, 128, 128, 128, 128, 128],
                                   # dense_layers=[128, 128],
                                   learning_rate=0.0005,
                                   momentum=0.9,
                                   epochs=100,
                                   batch_size=128,
                                   source=None,
                                   target=None
                                   )

        # TODO: either define a function to pass the nodes of the bottleneck targets or get them somewhere!
        X_cond_bn_train = [estimated_bn_fcts[idx, idx+1](train_samples[idx]) for idx in cond_idxs]
        y_bn_train = train_samples[target_idx]

        regr_bn.fit(X=np.concatenate([train_samples[source_idx],
                                      *X_cond_bn_train], axis=1),
                    y=y_bn_train)

        # Test on held out data
        X_cond_bn_test = [estimated_bn_fcts[idx, idx+1](test_samples[idx]) for idx in cond_idxs]
        y_hat_bn_test = regr_bn.predict(X=np.concatenate([test_samples[source_idx],
                                                          *X_cond_bn_test], axis=1))

        # y_hat_bn_test = estimated_mech_fcts[0, 2](estimated_bn_fcts[0, 2](test_samples[0])) + \
        #                 estimated_mech_fcts[1, 2](estimated_bn_fcts[1, 2](test_samples[1]))

        metrics_bn = {
            'mse': mean_squared_error(test_samples[target_idx], y_hat_bn_test),
            'mae': mean_absolute_error(test_samples[target_idx], y_hat_bn_test),
            'var': np.cov(test_samples[target_idx].T - y_hat_bn_test.T)
        }

        output_dict['bn'] = metrics_bn

    return output_dict


def main():
    ############### wandb section ###############
    # Can be ignored if not using wandb for experiment tracking
    wandb_config = dict(
        seed=None,
    )

    gettrace = getattr(sys, 'gettrace', None)

    if gettrace():
        print('Not using wandb for logging!')
        wandb_mode = 'offline'
    else:
        wandb_mode = 'online'

    wandb.init(
        entity='bings',
        project='bottlenecks',
        mode=wandb_mode,
        config=wandb_config
    )
    ##############################################

    GLOBAL_SEED = 0
    D_MICRO = 50
    N_BN_TRAIN = 20000
    N_TRAIN = 100
    N_TEST = 1000
    PREDICTORS = ['x', 'bn']
    MODE='linear'

    rs = np.random.RandomState(GLOBAL_SEED)
    seeds = rs.randint(low=0, high=1e5, size=10)

    # train_sample_sizes = [70, 75, 80, 85, 120, 200]
    train_sample_sizes = [100, 110, 150, 500, 1000]

    results_arr = np.empty(shape=(len(seeds), len(train_sample_sizes)),
                           dtype=object)
    sampler = SCBMSampler(seed=GLOBAL_SEED,
                          d_macro=3,
                          d_micro=D_MICRO,
                          d_bottleneck=2,
                          bottleneck_mode=MODE,
                          mech_mode=MODE,
                          p=0.99)

    # scbm = sampler.sample()

    # for i, seed in enumerate(seeds):
    #     # Sample an SCBM
    #     # sampler = SCBMSampler(seed=seed,
    #     #                       d_macro=3,
    #     #                       d_micro=50,
    #     #                       d_bottleneck=2,
    #     #                       bottleneck_mode='linear',
    #     #                       mech_mode='linear',
    #     #                       p=0.99)
    #     # scbm = sampler.sample()
    #     # define SCBM for given experiment
    #     # scbm = get_SCBM_tf_1(seed=seed, d=D_MICRO)

    for j, n_train in enumerate(train_sample_sizes):
        for i, seed, in enumerate(seeds):
            scbm = sampler.sample()
            # function that performs run for one setting
            results_arr[i, j] = single_transfer_run(seed=int(seed),
                                                    SCBM=scbm,
                                                    n_bn_train=N_BN_TRAIN,
                                                    n_train=n_train,
                                                    n_test=N_TEST,
                                                    target_idx=2,
                                                    source_idx=1,
                                                    cond_idxs=0,
                                                    cond_type=PREDICTORS,
                                                    mode=MODE)

    base_path = '/Users/Simon/Documents/PhD/Projects/CausalBottleneckModels/results'
    save_path = os.path.join(base_path, 'tf', MODE)

    np.save(os.path.join(save_path, f"{'_'.join(str(item) for item in train_sample_sizes)}_results.npy"), results_arr, allow_pickle=True)

    plot_multiple_transfer_runs(results_arr,
                                x_name='sample size',
                                x_values=train_sample_sizes,
                                y_name='mae',
                                predictors=PREDICTORS)

    a=0


if __name__ == '__main__':
    main()