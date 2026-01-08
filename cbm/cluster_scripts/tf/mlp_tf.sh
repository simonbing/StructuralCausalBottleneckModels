#! /usr/bin/bash

sbatch ../cluster_gpu.sh python ../../transfer_experiment.py \
--seed 0 --d_micro 10 --train_sample_sizes 10,20,50,500,1000 --n_bn_train 50000 --mode nonlinear \
--results_root /work/bd1083/b382081/projects/CausalBottleneckModels/results_2layer_swish