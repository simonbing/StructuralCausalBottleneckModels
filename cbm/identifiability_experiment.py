import os
import sys

from absl import flags, app
import numpy as np
import pandas as pd
import seaborn as sns
import wandb

from cbm.data import SCBMSampler
from cbm.estimation import estimate_bottleneck_and_mechanism_fcts
from cbm.eval import linear_bottleneck_eval, nonlinear_bottleneck_eval
from cbm.plotting import plot_multiple_bn_estimation_runs

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('n_seeds', 5, 'Number of repetitions of each setting.')
flags.DEFINE_enum('x', None, ['n_samples', 'd_macro', 'd_micro', 'd_bn'],
                  'Variable to experiment over with varying values (passed via x_values)')
flags.DEFINE_list('x_values', [], 'Range of values to perform experiments over.')
flags.DEFINE_integer('n_samples', 50000, 'Sample size.')
flags.DEFINE_integer('d_macro', 10, 'Number macro-variables.')
flags.DEFINE_integer('d_micro', 5, 'Number of micro-variables (per macro-variable).')
flags.DEFINE_integer('d_bn', 2, 'Dimension of bottleneck spaces.')
flags.DEFINE_string('estimation_mode', 'linear', 'Estimation mode.')
flags.DEFINE_enum('metric', 'r2', ['r2', 'mse'], 'Evaluation metric.')
flags.DEFINE_string('results_root',
                    '/Users/Simon/Documents/PhD/Projects/CausalBottleneckModels/results',
                    'Root path to results directory.')
flags.DEFINE_bool('save', False, 'Whether to save results.')


def single_bn_estimation_run(seed, n_samples, d_macro, d_micro, d_bn,
                             mode='linear', p=0.7, metric='r2'):
    match mode:
        case a if a in ('linear', 'reduced_rank'):
            bn_mech_mode = 'linear'
        case 'mlp':
            bn_mech_mode = 'nonlinear'

    sampler = SCBMSampler(seed=seed,
                          d_macro=d_macro,
                          d_micro=d_micro,
                          d_bottleneck=d_bn,
                          bottleneck_mode=bn_mech_mode,
                          mech_mode=bn_mech_mode,
                          p=p)
    # Sample SCBM
    SCBM = sampler.sample()

    # Sample from SCBM
    samples, bn_samples = SCBM.sample(size=n_samples)

    # Estimate bottleneck functions
    estimated_bn_fcts, _ = estimate_bottleneck_and_mechanism_fcts(SCBM=SCBM,
                                                                  samples=samples,
                                                                  mode=mode)

    # Apply learned bottlenecks
    n_vars = len(SCBM.variables)
    estimated_bn_samples = np.empty_like(estimated_bn_fcts, dtype=object)
    for i in range(n_vars):
        for j in range(n_vars):
            if estimated_bn_fcts[i, j] is not None:
                estimated_bn_samples[i, j] = estimated_bn_fcts[i, j](samples[i])

    # Evaluation
    match mode:
        case a if a in ('linear', 'reduced_rank'):
            eval_matrix = linear_bottleneck_eval(estimated_bn_samples,
                                                 SCBM.bottleneck_samples)
        case 'mlp':
            eval_matrix = nonlinear_bottleneck_eval(estimated_bn_samples,
                                                    bn_samples,
                                                    metric=metric)
    mean_score = np.mean(eval_matrix[eval_matrix != np.array(None)])

    return mean_score


def main(argv):
    ############### wandb section ###############
    # Can be ignored if not using wandb for experiment tracking
    wandb_config = dict(
        seed=FLAGS.seed,
        mode=FLAGS.estimation_mode,
        x=FLAGS.x,
        x_values=FLAGS.x_values,
        n_samples=FLAGS.n_samples,
        d_macro=FLAGS.d_macro,
        d_micro=FLAGS.d_micro,
        d_bn=FLAGS.d_bn,
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
    ##############################################

    results_path = os.path.join(FLAGS.results_root, 'id', FLAGS.estimation_mode,
                                f"{FLAGS.x}_{'_'.join(FLAGS.x_values)}")

    if os.path.exists(os.path.join(results_path, 'results.csv')):
        results = pd.read_csv(os.path.join(results_path, 'results.csv'))
    else:
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        rs = np.random.RandomState(FLAGS.seed)
        seeds = rs.randint(0, 1e5, size=FLAGS.n_seeds)

        x_names_list = []
        metrics_list = []

        for i, x_value in enumerate(FLAGS.x_values):
            for j, seed, in enumerate(seeds):
                run_args = {'seed': seed,
                            'n_samples': FLAGS.n_samples,
                            'd_macro': FLAGS.d_macro,
                            'd_micro': FLAGS.d_micro,
                            'd_bn': FLAGS.d_bn,
                            'mode': FLAGS.estimation_mode,
                            'metric': FLAGS.metric}
                # Change value of varying variable
                run_args[FLAGS.x] = int(x_value)

                x_names_list.append(x_value)
                metrics_list.append(single_bn_estimation_run(**run_args))

        results = pd.DataFrame({f'{FLAGS.x}': x_names_list, FLAGS.metric: metrics_list})

        # Save results dataframe
        results.to_csv(os.path.join(results_path, 'results.csv'))

    plot_multiple_bn_estimation_runs(results=results,
                                     x_name=FLAGS.x,
                                     y_name=FLAGS.metric,
                                     save=FLAGS.save,
                                     save_path=results_path)
    a=0


if __name__ == '__main__':
    app.run(main)