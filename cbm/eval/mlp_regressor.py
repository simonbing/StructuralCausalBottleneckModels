import copy

import jax.numpy as jnp
from flax import nnx
import optax
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import torch
from torch.utils.data import DataLoader
import wandb

from cbm.estimation.jax_utils import CBMDataset, numpy_collate


class MLPRegressor(object):
    def __init__(self, seed, d, dense_layers, learning_rate, momentum, epochs,
                 batch_size, source, target, d_out=None):
        self.seed = seed
        torch.manual_seed(self.seed)
        self.d = d
        self.d_out = d_out
        self.dense_layers = dense_layers
        # Build model
        self.model = self._build_model()

        self.epochs = epochs
        self.batch_size = batch_size

        self.optimizer = nnx.Optimizer(self.model, optax.adamw(learning_rate,
                                                               momentum))
        self.source = source
        self.target = target

    def _build_model(self):
        return MLP(d=self.d, dense_layers=self.dense_layers,
                   rngs=nnx.Rngs(params=self.seed), d_out=self.d_out)

    @staticmethod
    def loss_fn(model, X_batch, Y_batch):
        Y_hat_batch = model(X_batch)
        loss = jnp.sum((Y_hat_batch - Y_batch) ** 2, axis=1).mean()
        return loss

    @staticmethod
    @nnx.jit
    def train_step(model, optimizer, X_batch, Y_batch):
        grad_fn = nnx.value_and_grad(MLPRegressor.loss_fn, has_aux=False)
        loss, grads = grad_fn(model, X_batch, Y_batch)
        optimizer.update(grads)
        return loss

    @staticmethod
    @nnx.jit
    def eval_step(model, X_batch, Y_batch):
        loss_fn = MLPRegressor.loss_fn
        loss = loss_fn(model, X_batch, Y_batch)
        return loss

    def fit(self, X, y):
        # Split data into train and val sets
        X_train, X_val, Y_train, Y_val = train_test_split(X, y, train_size=0.8,
                                                          random_state=self.seed)

        train_dataloader = DataLoader(CBMDataset(X_train, Y_train),
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      collate_fn=numpy_collate)

        val_dataloader = DataLoader(CBMDataset(X_val, Y_val),
                                    batch_size=self.batch_size,
                                    shuffle=False,
                                    collate_fn=numpy_collate)

        # Train
        best_eval_loss = jnp.inf
        log_step_train = 0
        log_step_eval = 0
        for epoch in range(self.epochs):
            epoch_train_loss = 0
            for X_batch, Y_batch in train_dataloader:
                loss = self.train_step(self.model, self.optimizer, X_batch, Y_batch)
                epoch_train_loss += loss
                wandb.log({f'train loss ({self.source}, {self.target})': loss,
                           f'step_train ({self.source}, {self.target})': log_step_train})
                log_step_train += 1
            epoch_train_loss = epoch_train_loss / len(train_dataloader)
            wandb.log({f'epoch train loss ({self.source}, {self.target})': epoch_train_loss, f'epoch': epoch})
            # Eval
            epoch_eval_loss = 0
            for X_batch, Y_batch in val_dataloader:
                eval_loss = self.eval_step(self.model, X_batch, Y_batch)
                epoch_eval_loss += eval_loss
                wandb.log({f'eval loss ({self.source}, {self.target})': eval_loss,
                           f'step_eval ({self.source}, {self.target})': log_step_eval})
                log_step_eval += 1
            epoch_eval_loss = epoch_eval_loss / len(val_dataloader)
            wandb.log({f'epoch eval loss ({self.source}, {self.target})': epoch_eval_loss, f'epoch': epoch})
            if epoch_eval_loss < best_eval_loss:
                best_model = copy.deepcopy(self.model)
                best_eval_loss = epoch_eval_loss

        self.best_model = best_model

    @staticmethod
    @nnx.jit
    def prediction_step(model, X_batch):
        Y_hat_batch = model(X_batch)
        return Y_hat_batch

    def predict(self, X):
        score_dataloader = DataLoader(CBMDataset(X), batch_size=10000,
                                      shuffle=False, collate_fn=numpy_collate)
        # Get predictions
        Y_hat_list = []
        for X_batch in score_dataloader:
            Y_hat_batch = self.prediction_step(self.best_model, X_batch)
            Y_hat_list.append(Y_hat_batch)

        Y_hat = jnp.concatenate(Y_hat_list)

        return Y_hat

    def score(self, X, Y, metric='r2'):
        Y_hat = self.predict(X)

        match metric:
            case 'r2':
                score = r2_score(Y, Y_hat)
            case 'mse':
                score = ((Y_hat - Y) ** 2).mean()

        return score


class MLP(nnx.Module):
    def __init__(self, d, dense_layers, rngs, d_out=None):
        if not d_out:
            d_out = d

        layers_MLP = []
        for i in range(len(dense_layers)):
            if i == 0:
                layers_MLP.append(nnx.Linear(d, dense_layers[i], rngs=rngs))
            else:
                layers_MLP.append(nnx.Linear(dense_layers[i-1], dense_layers[i],
                                             rngs=rngs))
            layers_MLP.append(nnx.swish)
        layers_MLP.append(nnx.Linear(dense_layers[-1], d_out, rngs=rngs))
        self.mlp = nnx.Sequential(*layers_MLP)

    def __call__(self, x):
        y_hat = self.mlp(x)
        return y_hat
