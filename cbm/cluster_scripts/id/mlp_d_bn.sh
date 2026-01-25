#! /usr/bin/bash

sbatch ../cluster_gpu.sh python ../../identifiability_experiment.py \
--seed 42 --x d_bn --x_values 10 --d_micro 100 --d_macro 3 --n_samples 50000 --estimation_mode mlp \
--results_root /work/bd1083/b382081/projects/CausalBottleneckModels/results_2layer_swish_deep_mlp_debug_1