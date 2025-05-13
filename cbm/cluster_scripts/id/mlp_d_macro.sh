#! /usr/bin/bash

sbatch ../cluster_gpu.sh python ../../identifiability_experiment.py \
--seed 42 --x d_macro --x_values 20 --estimation_mode mlp \
--results_root /work/bd1083/b382081/projects/CausalBottleneckModels/results