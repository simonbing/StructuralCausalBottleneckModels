#! /usr/bin/bash

sbatch ../cluster_gpu.sh python ../../identifiability_experiment.py \
--x n_samples --x_values 1000,10000,30000,50000,100000 --estimation_mode mlp \
--results_root /work/bd1083/b382081/projects/CausalBottleneckModels/results_2layer_swish_deep_mlp