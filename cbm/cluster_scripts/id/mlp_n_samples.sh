#! /usr/bin/bash

sbatch ../cluster_gpu.sh python ../../identifiability_experiment.py \
--x n_samples --x_values 100000 --estimation_mode mlp --seed 0 --n_seeds 2 \
--results_root /work/bd1083/b382081/projects/CausalBottleneckModels/results_2layer_swish_deep_mlp_pt1