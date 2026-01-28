# Structural Causal Bottleneck Models

Official code repository for the paper "Structural Causal Bottleneck Models"

### Getting Started
We implement our bottleneck estimation pipeline using the JAX ML library. To start, install all required dependecies by running

```bash
pip install -r requirements.txt
```

## Experiments
We provide self-contained scripts to reproduce all of our experiments. Instructions on how to run them are provided below. 

We use the ```wandb``` library to track experiments, but this is optional and can be turned off by setting

```bash
export WANDB_MODE=disabled
```

General flags shared across scripts are

| Flag | Description |
|------|-------------|
| `--estimation_mode` | [`linear`, `mlp`] Switch between linear and nonlinear models.|
| `--n_samples` | Number of training samples. |
| `--d_macro` | Number of nodes in graph. |
| `--d_micro` | Number of internal dimensions per node. |
| `--d_bn` | Bottleneck dimension. |
| `--n_seeds` | Number of (randomly re-initialised) repetitions per setting. |

### Identifiability

```bash
python identifiability_experiment.py --x n_samples --x_values 100,1000,10000,30000 --estimation_mode linear --n_seeds 5 --results_root /path/to/save/results
```

The flag `--x` indicates which value to vary and `--x_values` denotes the values to cover.

### Misspecification
Linear 
```bash
python misspecification_experiment.py --n_samples 30000 --d_macro 3 --d_micro 100 --true_d_bn 10 --p 1.0 --d_bn_values 1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50 --n_seeds 5 --results_root /path/to/save/results
```
Nonlinear
```bash
python misspecification_experiment_nonlinear.py --n_samples 50000 --d_macro 3 --d_micro 100 --true_d_bn 10 --p 1.0 --d_bn_values 1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50 --n_seeds 5 --results_root /path/to/save/results
```

`--true_d_bn` indicates the ground-truth bottleneck dimension here and `--d_bn_values` the values which to consider as the assumed bottleneck dimension.

### Transfer Learning

```bash
python transfer_experiment.py --d_micro 10 --train_sample_sizes 10,20,50,500,1000 --mode nonlinear --results_root /path/to/save/results
```

## License

The code in this repository is shared under the permissive [MIT license](https://opensource.org/license/mit/). A copy of can be found in [LICENSE](LICENSE).