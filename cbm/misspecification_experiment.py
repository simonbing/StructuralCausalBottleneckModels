"""
Idea:
 - Identifiability experiment, but with misspecified bottleneck dimension
 - Ideally: see that the correct bottleneck dimension is a lower bound. Lower and we get worst results, higher and we are still fine
 - Nonlinear case: possibly our reconstruction loss starts to decrease even in training -> unsupervised check!
 - Plot metric vs assumed bottleneck dimension: want to see something like an elbow at the true bottleneck dimension
 - How does the metric change when one thing has different size? Does the r2 still work?

 First try: linear, two nodes, d_x = 10, d_z = 2 or 3
"""

import os
import sys

from absl import app, flags
import numpy as np

from cbm.data import SCBMSampler
from cbm.estimation import estimate_bottleneck_and_mechanism_fcts
from cbm.eval import linear_bottleneck_eval, nonlinear_bottleneck_eval

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('n_samples', 30000, 'Sample size.')
flags.DEFINE_integer('d_macro', 10, 'Number macro-variables.')
flags.DEFINE_integer('d_micro', 5, 'Number of micro-variables (per macro-variable).')
flags.DEFINE_integer('true_d_bn', 2, 'True dimension of bottleneck spaces.')
flags.DEFINE_integer('assumed_d_bn', 2, 'Assumed dimension of bottleneck spaces.')
flags.DEFINE_float('p', 0.7, 'Connection probability of SCBM.')
flags.DEFINE_enum('metric', 'r2', ['r2', 'mse'], 'Evaluation metric.')
flags.DEFINE_string('estimation_mode', 'linear', 'Estimation mode.')

def single_misspecification_run(seed, n_samples, d_macro, d_micro, true_d_bn, assumed_d_bn,
                                mode='linear', p=0.7, metric='r2'):
    # I think the main difference to exisiting stuff: adapt estimation to explicitly take assumed_d_bn as input
    
    match mode:
        case a if a in ('linear', 'reduced_rank'):
            bn_mech_mode = 'linear'
        case 'mlp':
            bn_mech_mode = 'nonlinear'

    sampler = SCBMSampler(seed=seed,
                          d_macro=d_macro,
                          d_micro=d_micro,
                          d_bottleneck=true_d_bn,
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
                                                                  mode=mode,
                                                                  assumed_d_bn=assumed_d_bn)
    
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
        true_d_bn=FLAGS.true_d_bn,
        assumed_d_bn=FLAGS.assumed_d_bn,
        p=FLAGS.p,
        metric=FLAGS.metric
    )

    wandb.init(
        entity='bings',
        project='bottlenecks',
        config=wandb_config
    )
    ############### End wandb section ###############

    mean_score = single_misspecification_run(
        seed=FLAGS.seed,
        n_samples=FLAGS.n_samples,
        d_macro=FLAGS.d_macro,
        d_micro=FLAGS.d_micro,
        true_d_bn=FLAGS.true_d_bn,
        assumed_d_bn=FLAGS.assumed_d_bn,
        mode=FLAGS.estimation_mode,
        p=FLAGS.p,
        metric=FLAGS.metric
    )

    print(f'Mean score: {mean_score}')
    wandb.log({'mean_score': mean_score})

if __name__ == '__main__':
    app.run(main)