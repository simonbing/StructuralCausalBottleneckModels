#! /usr/bin/bash

sbatch ../cluster_gpu.sh python ../../identifiability_experiment.py \
--seed 2 --x d_macro --x_values 2 --d_micro 100 --d_bn 10 --n_samples 50000 --n_seeds 5 --estimation_mode mlp \
--results_root /work/bd1083/b382081/projects/CausalBottleneckModels/results_2layer_swish_deep_mlp_d_macro_3_extra