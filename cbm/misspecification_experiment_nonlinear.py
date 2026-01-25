import os
import sys

from absl import app, flags
import jax
import numpy as np
import pandas as pd
import wandb

from cbm.data import SCBMSampler
from cbm.estimation import estimate_bottleneck_and_mechanism_fcts
from cbm.estimation.ae_regressor import AutoencoderRegressor
from cbm.eval import linear_bottleneck_eval, nonlinear_bottleneck_eval
from cbm.plotting import plot_multiple_misspecifcation_runs

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('n_seeds', 5, 'Number of repetitions of each setting.')
flags.DEFINE_integer('n_samples', 30000, 'Sample size.')
flags.DEFINE_integer('d_macro', 3, 'Number macro-variables.')
flags.DEFINE_integer('d_micro', 100, 'Number of micro-variables (per macro-variable).')
flags.DEFINE_integer('true_d_bn', 10, 'True dimension of bottleneck spaces.')
flags.DEFINE_integer('assumed_d_bn', 2, 'Assumed dimension of bottleneck spaces.')
flags.DEFINE_list('d_bn_values', [], 'Range of values to perform experiments over.')
flags.DEFINE_float('p', 1.0, 'Connection probability of SCBM.')
flags.DEFINE_enum('metric', 'r2', ['r2', 'mse'], 'Evaluation metric.')
flags.DEFINE_string('estimation_mode', 'linear', 'Estimation mode.')
flags.DEFINE_string('results_root',
                    '/Users/Simon/Documents/PhD/Projects/CausalBottleneckModels/results',
                    'Root path to results directory.')

def estimate_specific_bottleneck_fcts(SCBM, samples, assumed_d_bn,
                                       mode='mlp'):
    """
    Estimate bottleneck Z_{(2, 3)} for a three node SCBM (with misspecified d_bn)
    """
    reg_model = AutoencoderRegressor

    # Estimate bottleneck 1 -> 2
    source = SCBM.variables[0]
    target = SCBM.variables[1]

    # No conditioning set needed for this estimation

    reg_args = {'seed': SCBM.seed,
                'd_micro_in': source.d,
                'd_micro_out': target.d,
                'd_bottleneck': SCBM.d_bottleneck_matrix[0, 1],  # use true d_bn here
                'source': 0,
                'target': 1,
                'd_cond': 0,
                'dense_x_z': [256, 128, 64, 64, 32],
                'dense_z_x': [32, 64, 64, 128, 256],
                'epochs': 500,
                'batch_size': 1024,
                'learning_rate': 0.00001,
                'momentum': 0.9}
    regressor = reg_model(**reg_args)

    regressor.fit(X=samples[0],
                  Y=samples[1],
                  X_cond=[])
    
    bottleneck_fct_1_2, _ = regressor.get_bottleneck_and_mechanism_fcts()

    # Estimate bottleneck 2 -> 3
    source = SCBM.variables[1]
    target = SCBM.variables[2]

    cond_set = np.concatenate([bottleneck_fct_1_2(samples[0])], axis=1)

    reg_args = {'seed': SCBM.seed,
                'd_micro_in': source.d,
                'd_micro_out': target.d,
                'd_bottleneck': assumed_d_bn,  # use assumed, misspecified d_bn
                'source': 1,
                'target': 2,
                'd_cond': cond_set.shape[1],
                'dense_x_z': [256, 128, 64, 64, 32],
                'dense_z_x': [32, 64, 64, 128, 256],
                'epochs': 500,
                'batch_size': 1024,
                'learning_rate': 0.00001,
                'momentum': 0.9}
    regressor = reg_model(**reg_args)

    regressor.fit(X=samples[1],
                  Y=samples[2],
                  X_cond=cond_set)

    bottleneck_fct_2_3, _ = regressor.get_bottleneck_and_mechanism_fcts()

    return bottleneck_fct_2_3


def single_nonlin_misspecification_run(scbm, scbm_samples, scbm_bn_samples, assumed_d_bn,
                                mode='linear', p=0.7, metric='r2'):

    # Estimate bottleneck function 2 -> 3
    bottleneck_fct_2_3 = estimate_specific_bottleneck_fcts(scbm, scbm_samples, assumed_d_bn)

    # Apply learned bottlenecks
    estimated_bn_samples = np.empty_like(scbm_bn_samples, dtype=object)

    estimated_bn_samples[1, 2] = bottleneck_fct_2_3(scbm_samples[1])

    # Evaluation
    eval_matrix = nonlinear_bottleneck_eval(estimated_bn_samples,
                                            scbm_bn_samples,
                                            metric=metric)
    mean_score = np.mean(eval_matrix[eval_matrix != np.array(None)])

    return mean_score

def main(argv):
    ############### wandb section ###############
    # Can be ignored if not using wandb for experiment tracking
    wandb_config = dict(
        seed=FLAGS.seed,
        mode=FLAGS.estimation_mode,
        n_samples=FLAGS.n_samples,
        d_macro=FLAGS.d_macro,
        d_micro=FLAGS.d_micro,
        true_d_bn=FLAGS.true_d_bn,
        assumed_d_bn=FLAGS.assumed_d_bn,
        p=FLAGS.p,
        metric=FLAGS.metric
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
    ############### End wandb section ###############

    jax.config.update("jax_default_matmul_precision", "float32")

    results_path = os.path.join(FLAGS.results_root,
                                'misspecification_id',
                                FLAGS.estimation_mode,
                                f"assumed_d_bn_{'_'.join(FLAGS.d_bn_values)}",
                                f'{FLAGS.seed}')
    
    if os.path.exists(os.path.join(results_path, 'results.csv')):
        results = pd.read_csv(os.path.join(results_path, 'results.csv'))
    else:
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        rs = np.random.RandomState(FLAGS.seed)
        seeds = rs.randint(0, 1e5, size=FLAGS.n_seeds)

        # Sample data for all seeds and cache
        scbm_cache = {}
        match FLAGS.estimation_mode:
            case a if a in ('linear', 'reduced_rank'):
                bn_mech_mode = 'linear'
            case 'mlp':
                bn_mech_mode = 'nonlinear'

        for seed in seeds:
            sampler = SCBMSampler(seed=seed,
                                  d_macro=FLAGS.d_macro,
                                  d_micro=FLAGS.d_micro,
                                  d_bottleneck=FLAGS.true_d_bn,
                                  bottleneck_mode=bn_mech_mode,
                                  mech_mode=bn_mech_mode,
                                  p=FLAGS.p)

            scbm = sampler.sample()

            samples, bn_samples = scbm.sample(size=FLAGS.n_samples)
            scbm_cache[seed] = {'scbm': scbm, 'samples': samples, 'bn_samples': bn_samples}

        assumed_d_bn_list = []
        mean_scores = []

        for assumed_d_bn in FLAGS.d_bn_values:
            for seed in seeds:
                run_args = {
                    'scbm': scbm_cache[seed]['scbm'],
                    'scbm_samples': scbm_cache[seed]['samples'],
                    'scbm_bn_samples': scbm_cache[seed]['bn_samples'],
                    'assumed_d_bn': int(assumed_d_bn),
                    'mode': FLAGS.estimation_mode,
                    'p': FLAGS.p,
                    'metric': FLAGS.metric
                }

                assumed_d_bn_list.append(assumed_d_bn)
                mean_scores.append(single_nonlin_misspecification_run(**run_args))

                a=0

        results = pd.DataFrame({'assumed_d_bn': assumed_d_bn_list,
                                FLAGS.metric: mean_scores})
        
        # Print results
        print(results)

        # Save results dataframe
        results.to_csv(os.path.join(results_path, 'results.csv'))

    # Plot results
    plot_multiple_misspecifcation_runs(results=results, x_name='assumed_d_bn', y_name=FLAGS.metric,
                                       true_d_bn=FLAGS.true_d_bn, save=True, save_path=results_path)
    
    a=0

if __name__ == '__main__':
    app.run(main)