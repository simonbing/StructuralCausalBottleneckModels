import sys
import os

from absl import flags, app
import jax
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

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('n_seeds', 10, 'Number of repetitions of each setting.')
flags.DEFINE_enum('mode', 'linear', ['linear', 'nonlinear'], 'Estimation mode.')
flags.DEFINE_integer("d_micro", 50, "Number of micro-variables (per macro-variable).")
flags.DEFINE_integer('n_bn_train', 20000, 'Number of samples for bottleneck training.')
flags.DEFINE_integer('n_test', 1000, 'Number of test samples for transfer task.')
flags.DEFINE_list('train_sample_sizes', [], 'Training sample size range.')
flags.DEFINE_string('results_root',
                    '',
                    'Root path to results directory.')


def single_transfer_run(seed, SCBM, n_bn_train, n_train, n_test, target_idx,
                        source_idx, cond_idxs, estimated_bn_fcts, cond_type=['x', 'bn'],
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

        if mode == 'linear':
            regr_bn = LinearRegression()
        elif mode == 'nonlinear':
            regr_bn = MLPRegressor(seed=seed,
                                   d=12,  # TODO: automatically calculate this!
                                   d_out=train_samples[0].shape[-1],
                                   dense_layers=[128, 128, 128, 128, 128, 128],
                                   learning_rate=0.0005,
                                   momentum=0.9,
                                   epochs=100,
                                   batch_size=128,
                                   source=None,
                                   target=None
                                   )

        X_cond_bn_train = [estimated_bn_fcts[idx, idx+1](train_samples[idx]) for idx in cond_idxs]
        y_bn_train = train_samples[target_idx]

        regr_bn.fit(X=np.concatenate([train_samples[source_idx],
                                      *X_cond_bn_train], axis=1),
                    y=y_bn_train)

        # Test on held out data
        X_cond_bn_test = [estimated_bn_fcts[idx, idx+1](test_samples[idx]) for idx in cond_idxs]
        y_hat_bn_test = regr_bn.predict(X=np.concatenate([test_samples[source_idx],
                                                          *X_cond_bn_test], axis=1))

        metrics_bn = {
            'mse': mean_squared_error(test_samples[target_idx], y_hat_bn_test),
            'mae': mean_absolute_error(test_samples[target_idx], y_hat_bn_test),
            'var': np.cov(test_samples[target_idx].T - y_hat_bn_test.T)
        }

        output_dict['bn'] = metrics_bn

    return output_dict


def main(argv):
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
        entity='wandbusername',
        project='wandbproject',
        mode=wandb_mode,
        config=wandb_config
    )
    ##############################################

    # Debug
    jax.config.update("jax_default_matmul_precision", "float32")

    PREDICTORS = ['x', 'bn']

    # Debug
    jax.config.update("jax_default_matmul_precision", "float32")

    # Check if gpu is being used
    print(f'Using device: {jax.default_backend()}')

    results_path = os.path.join(FLAGS.results_root,
                                'tf',
                                FLAGS.mode,
                                f'seed_{FLAGS.seed}')
    
    results_arr_path = os.path.join(results_path, f"{'_'.join(str(item) for item in FLAGS.train_sample_sizes)}_results.npy")

    if os.path.exists(results_arr_path):
        print('Loading existing results array...')

        results_arr = np.load(results_arr_path, allow_pickle=True)
    else:
        print('No existing results array found, running experiments...')

        if not os.path.exists(results_path):
            os.makedirs(results_path)

        rs = np.random.RandomState(FLAGS.seed)
        seeds = rs.randint(low=0, high=1e5, size=FLAGS.n_seeds)

        FLAGS.train_sample_sizes = [int(size) for size in FLAGS.train_sample_sizes]

        results_arr = np.empty(shape=(len(seeds), len(FLAGS.train_sample_sizes)),
                            dtype=object)
        sampler = SCBMSampler(seed=FLAGS.seed,
                            d_macro=3,
                            d_micro=FLAGS.d_micro,
                            d_bottleneck=2,
                            bottleneck_mode=FLAGS.mode,
                            mech_mode=FLAGS.mode,
                            p=0.99)

        bottlenecks_cache = [None] * len(seeds)

        for j, n_train in enumerate(FLAGS.train_sample_sizes):
            for i, seed, in enumerate(seeds):
                scbm = sampler.sample()

                bn_train_samples, bn_train_bn_samples = scbm.sample(size=FLAGS.n_bn_train)

                if bottlenecks_cache[i] is None:
                    bn_mode = 'linear' if FLAGS.mode == 'linear' else 'mlp'
                    estimated_bn_fcts, _ = estimate_bottleneck_and_mechanism_fcts(scbm, bn_train_samples, mode=bn_mode)
                    bottlenecks_cache[i] = estimated_bn_fcts

                results_arr[i, j] = single_transfer_run(seed=int(seed),
                                                        SCBM=scbm,
                                                        n_bn_train=FLAGS.n_bn_train,
                                                        n_train=n_train,
                                                        n_test=FLAGS.n_test,
                                                        target_idx=2,
                                                        source_idx=1,
                                                        cond_idxs=0,
                                                        estimated_bn_fcts=bottlenecks_cache[i],
                                                        cond_type=PREDICTORS,
                                                        mode=FLAGS.mode)
                
            # Print intermediate results
            x_mae = np.mean([results_arr[idx, j]['x']['mae'] for idx in range(results_arr.shape[0])])
            bn_mae = np.mean([results_arr[idx, j]['bn']['mae'] for idx in range(results_arr.shape[0])])
            print(f"Train size {n_train}:\n x_mae  = {x_mae:.7f}\n bn_mae = {bn_mae:.7f}")



        np.save(results_arr_path, results_arr, allow_pickle=True)

    plot_multiple_transfer_runs(results_arr,
                                x_name='sample size',
                                x_values=FLAGS.train_sample_sizes,
                                y_name='mae',
                                predictors=PREDICTORS)

if __name__ == '__main__':
    app.run(main)