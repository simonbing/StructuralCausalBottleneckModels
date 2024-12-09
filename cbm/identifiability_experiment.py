import os

from absl import flags, app
import numpy as np

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
flags.DEFINE_integer('d_macro', 3, 'Number macro-variables.')
flags.DEFINE_integer('d_micro', 5, 'Number of micro-variables (per macro-variable).')
flags.DEFINE_integer('d_bn', 2, 'Dimension of bottleneck spaces.')
flags.DEFINE_string('estimation_mode', 'linear', 'Estimation mode.')
flags.DEFINE_string('results_root',
                    '/Users/Simon/Documents/PhD/Projects/CausalBottleneckModels/results',
                    'Root path to results directory.')
flags.DEFINE_bool('save', False, 'Whether to save results.')


def single_bn_estimation_run(seed, n_samples, d_macro, d_micro, d_bn,
                             mode='linear', p=0.7):
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
                                                    bn_samples)
    mean_score = np.mean(eval_matrix[eval_matrix != np.array(None)])

    return mean_score


def main(argv):
    results_path = os.path.join(FLAGS.results_root, 'id', FLAGS.estimation_mode,
                                f"{FLAGS.x}_{'_'.join(FLAGS.x_values)}")

    if os.path.exists(os.path.join(results_path, 'results.npy')):
        results = np.load(os.path.join(results_path, 'results.npy'))
    else:
        os.makedirs(results_path)

        rs = np.random.RandomState(FLAGS.seed)
        seeds = rs.randint(0, 1e6, size=FLAGS.n_seeds)

        results = np.empty((FLAGS.n_seeds, len(FLAGS.x_values)))
        for i, x_value in enumerate(FLAGS.x_values):
            for j, seed, in enumerate(seeds):
                run_args = {'seed': seed,
                            'n_samples': FLAGS.n_samples,
                            'd_macro': FLAGS.d_macro,
                            'd_micro': FLAGS.d_micro,
                            'd_bn': FLAGS.d_bn,
                            'mode': FLAGS.estimation_mode}
                # Change value of varying variable
                run_args[FLAGS.x] = int(x_value)

                results[j, i] = single_bn_estimation_run(**run_args)

        # Save results matrix
        np.save(os.path.join(results_path, 'results.npy'), results)

    plot_multiple_bn_estimation_runs(results=results,
                                     x_name=FLAGS.x,
                                     x_values=FLAGS.x_values,
                                     y_name='r2' if FLAGS.estimation_mode=='linear' else 'Acc.',
                                     save=FLAGS.save,
                                     save_path=results_path)
    a=0


if __name__ == '__main__':
    app.run(main)