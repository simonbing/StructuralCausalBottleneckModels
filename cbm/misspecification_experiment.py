import os
import sys

from absl import app, flags
import jax
import numpy as np
import pandas as pd
import wandb

from cbm.data import SCBMSampler
from cbm.estimation import estimate_bottleneck_and_mechanism_fcts
from cbm.eval import linear_bottleneck_eval, nonlinear_bottleneck_eval
from cbm.plotting import plot_multiple_misspecifcation_runs

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('n_seeds', 5, 'Number of repetitions of each setting.')
flags.DEFINE_integer('n_samples', 30000, 'Sample size.')
flags.DEFINE_integer('d_macro', 10, 'Number macro-variables.')
flags.DEFINE_integer('d_micro', 5, 'Number of micro-variables (per macro-variable).')
flags.DEFINE_integer('true_d_bn', 2, 'True dimension of bottleneck spaces.')
flags.DEFINE_integer('assumed_d_bn', 2, 'Assumed dimension of bottleneck spaces.')
flags.DEFINE_list('d_bn_values', [], 'Range of values to perform experiments over.')
flags.DEFINE_float('p', 0.7, 'Connection probability of SCBM.')
flags.DEFINE_enum('metric', 'r2', ['r2', 'mse'], 'Evaluation metric.')
flags.DEFINE_string('estimation_mode', 'linear', 'Estimation mode.')
flags.DEFINE_string('results_root',
                    '',
                    'Root path to results directory.')

def single_misspecification_run(scbm, scbm_samples, scbm_bn_samples, assumed_d_bn,
                                mode='linear', p=0.7, metric='r2'):
    # Estimate bottleneck functions
    estimated_bn_fcts, _ = estimate_bottleneck_and_mechanism_fcts(SCBM=scbm,
                                                                  samples=scbm_samples,
                                                                  mode=mode,
                                                                  assumed_d_bn=assumed_d_bn)
    
    # Apply learned bottlenecks
    n_vars = len(scbm.variables)
    estimated_bn_samples = np.empty_like(estimated_bn_fcts, dtype=object)
    for i in range(n_vars):
        for j in range(n_vars):
            if estimated_bn_fcts[i, j] is not None:
                estimated_bn_samples[i, j] = estimated_bn_fcts[i, j](scbm_samples[i])

    # Evaluation
    match mode:
        case a if a in ('linear', 'reduced_rank'):
            eval_matrix = linear_bottleneck_eval(estimated_bn_samples,
                                                 scbm_bn_samples,
                                                 )
        case 'mlp':
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
        entity='wandbusername',
        project='wandbproject',
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
                mean_scores.append(single_misspecification_run(**run_args))

        results = pd.DataFrame({'assumed_d_bn': assumed_d_bn_list,
                                FLAGS.metric: mean_scores})
        
        # Print results
        print(results)

        # Save results dataframe
        results.to_csv(os.path.join(results_path, 'results.csv'))

    # Plot results
    plot_multiple_misspecifcation_runs(results=results, x_name='assumed_d_bn', y_name=FLAGS.metric,
                                       true_d_bn=FLAGS.true_d_bn, save=True, save_path=results_path)
    
if __name__ == '__main__':
    app.run(main)