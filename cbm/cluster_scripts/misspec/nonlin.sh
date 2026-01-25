#! /usr/bin/bash

sbatch ../cluster_gpu.sh python ../../misspecification_experiment.py \
--seed 0 --n_seeds 5 --n_samples 30000 --d_macro 2 --d_micro 50 --true_d_bn 10 --p 1.0 --metric r2 \
--d_bn_values 1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50 --estimation_mode mlp \
--results_root /work/bd1083/b382081/projects/CausalBottleneckModels/results_4layer_relu