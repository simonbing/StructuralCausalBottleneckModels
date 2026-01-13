#! /usr/bin/bash

sbatch ../cluster_gpu.sh python ../../identifiability_experiment.py \
--x d_micro --x_values 5 --n_samples 50000 --estimation_mode mlp \
--results_root /work/bd1083/b382081/projects/CausalBottleneckModels/results_2layer_swish_deep_mlp