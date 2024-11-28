import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

from cbm.estimation import estimate_bottleneck_and_mechanism_fcts
from cbm.SCBM_models import get_SCBM_tf_1
from cbm.utils import make_iterable
from cbm.plotting import plot_multiple_runs


def single_transfer_run(seed, SCBM, n_bn_train, n_train, n_test, target_idx,
                        source_idx, cond_idxs, predictors=['ols', 'bn']):
    cond_idxs = make_iterable(cond_idxs)

    train_samples, train_bn_samples = SCBM.sample(size=n_train)
    test_samples, test_bn_samples = SCBM.sample(size=n_test)

    output_dict = {}

    if 'ols' in predictors:
        # Fit OLS without applying bottlenecks
        regr_ols = LinearRegression()

        X_cond_ols_train = [train_samples[idx] for idx in cond_idxs]
        y_ols_train = train_samples[target_idx]

        regr_ols.fit(X=np.concatenate([train_samples[source_idx],
                                       *X_cond_ols_train], axis=1),
                     y=y_ols_train)

        # Test on held out data
        X_cond_ols_test = [test_samples[idx] for idx in cond_idxs]
        y_hat_ols_test = regr_ols.predict(X=np.concatenate([test_samples[source_idx],
                                                            *X_cond_ols_test], axis=1))

        metrics = {
            'mse': mean_squared_error(test_samples[target_idx], y_hat_ols_test),
            'mae': mean_absolute_error(test_samples[target_idx], y_hat_ols_test),
            'var': np.cov(test_samples[target_idx].T - y_hat_ols_test.T)
        }

        output_dict['ols'] = metrics

    if 'bn' in predictors:
        # Fit OLS with bottlenecks applied to conditioning set

        # Estimate bottleneck functions
        bn_train_samples, bn_train_bn_samples = SCBM.sample(size=n_bn_train)
        estimated_bn_fcts, _ = estimate_bottleneck_and_mechanism_fcts(SCBM,
                                                                      bn_train_samples,
                                                                      mode='linear')
        regr_bn = LinearRegression()

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

        metrics = {
            'mse': mean_squared_error(test_samples[target_idx], y_hat_bn_test),
            'mae': mean_absolute_error(test_samples[target_idx], y_hat_bn_test),
            'var': np.cov(test_samples[target_idx].T - y_hat_bn_test.T)
        }

        output_dict['bn'] = metrics

    return output_dict


def main():
    GLOBAL_SEED = 0
    D_MICRO = 80
    N_BN_TRAIN = 20000
    N_TRAIN = 100
    N_TEST = 1000
    PREDICTORS = ['ols', 'bn']

    rs = np.random.RandomState(GLOBAL_SEED)
    seeds = rs.randint(low=0, high=1e5, size=5)

    train_sample_sizes = [100, 200, 500, 1000, 10000]

    results_arr = np.empty(shape=(len(seeds), len(train_sample_sizes)),
                           dtype=object)
    for i, seed in enumerate(seeds):
        # define SCBM for given experiment
        scbm = get_SCBM_tf_1(seed=seed, d=D_MICRO)

        output_list = []
        for j, n_train in enumerate(train_sample_sizes):
            # function that performs run for one setting
            results_arr[i, j] = single_transfer_run(seed=seed,
                                                    SCBM=scbm,
                                                    n_bn_train=N_BN_TRAIN,
                                                    n_train=n_train,
                                                    n_test=N_TEST,
                                                    target_idx=2,
                                                    source_idx=1,
                                                    cond_idxs=0,
                                                    predictors=PREDICTORS)

    plot_multiple_runs(results_arr,
                       x_name='sample size',
                       x_values=train_sample_sizes,
                       y_name='mae',
                       predictors=PREDICTORS)

    a=0


if __name__ == '__main__':
    main()