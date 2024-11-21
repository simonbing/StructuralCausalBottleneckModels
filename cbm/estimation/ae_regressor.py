import copy

import jax
import jax.numpy as jnp
from flax import nnx
import optax
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import wandb

### DEBUG
from cbm.eval.mlp_regressor import MLP
#########

from cbm.estimation.base_regressor import BaseRegressor
from cbm.estimation.jax_models import Autoencoder
from cbm.estimation.jax_utils import CBMDataset, numpy_collate


class AutoencoderRegressor(BaseRegressor):
    def __init__(self, seed, d_micro_in, d_micro_out, d_bottleneck, source,
                 target, d_cond, dense_x_z, dense_z_x, epochs, batch_size,
                 learning_rate, momentum):
        super().__init__(seed, d_micro_in, d_micro_out, d_bottleneck, source,
                         target, d_cond)
        torch.manual_seed(self.seed)

        self.model = Autoencoder(in_dim=self.d_micro_in, dense_x_z=dense_x_z,
                                 dense_z_x=dense_z_x, z_dim=self.d_bottleneck,
                                 cond_dim=self.d_cond, out_dim=self.d_micro_out,
                                 rngs=nnx.Rngs(params=self.seed))

        # self.model = MLP(d=self.d_micro_in, dense_layers=[128, 128, 128, 128],
        #                  rngs=nnx.Rngs(params=self.seed))

        self.epochs = epochs
        self.batch_size = batch_size

        self.optimizer = nnx.Optimizer(self.model, optax.adamw(learning_rate,
                                                               momentum))

    def fit(self, X, Y, X_cond=[]):
        if len(X_cond) != 0:
            assert self.d_cond != 0, 'Must pass a conditioning set if d_cond != 0!'
        # Split data
        if len(X_cond) == 0:
            X_train, X_val, Y_train, Y_val = train_test_split(X, Y,
                                                              train_size=0.8,
                                                              random_state=self.seed)
        else:
            X_train, X_val, X_cond_train, X_cond_val, Y_train, Y_val \
                = train_test_split(X, X_cond, Y, train_size=0.8, random_state=self.seed)
        # Build dataloaders
        if len(X_cond) == 0:
            train_data = [X_train, Y_train]
            val_data = [X_val, Y_val]

        else:
            train_data = [X_train, X_cond_train, Y_train]
            val_data = [X_val, X_cond_val, Y_val]

        train_dataloader = DataLoader(CBMDataset(*train_data),
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      collate_fn=numpy_collate)
        val_dataloader = DataLoader(CBMDataset(*val_data),
                                    batch_size=self.batch_size,
                                    shuffle=False,
                                    collate_fn=numpy_collate)

        # Train
        best_eval_loss = jnp.inf
        log_step_train = 0
        log_step_eval = 0
        for epoch in range(self.epochs):
            for batch in train_dataloader:
                source_batch = batch[:-1]  # also includes X_cond if it's there
                target_batch = batch[-1]
                loss = self.train_step(self.model, self.optimizer, source_batch, target_batch)
                wandb.log({f'train loss ({self.source}, {self.target}), estim': loss,
                           f'step_train ({self.source}, {self.target}), estim': log_step_train})
                log_step_train += 1
            # Eval
            epoch_eval_loss = 0
            for batch in val_dataloader:
                source_batch = batch[:-1]  # also includes X_cond if it's there
                target_batch = batch[-1]
                eval_loss = self.eval_step(self.model, source_batch, target_batch)
                epoch_eval_loss += eval_loss
                wandb.log({f'eval loss ({self.source}, {self.target}), estim': eval_loss,
                           f'step_eval ({self.source}, {self.target}), estim': log_step_eval})
                log_step_eval += 1
            epoch_eval_loss = epoch_eval_loss / len(val_dataloader)
            if epoch_eval_loss < best_eval_loss:
                best_model = copy.deepcopy(self.model)
                best_eval_loss = eval_loss

        self.best_model = best_model

        a = 0

    @staticmethod
    def loss_fn(model, source_batch, target_batch):
        out_batch = model(*source_batch)
        loss = jnp.sum((out_batch - target_batch) ** 2, axis=1).mean()
        return loss

    @staticmethod
    @nnx.jit
    def train_step(model, optimizer, source_batch, target_batch):
        grad_fn = nnx.value_and_grad(AutoencoderRegressor.loss_fn, has_aux=False)
        loss, grads = grad_fn(model, source_batch, target_batch)
        optimizer.update(grads)
        return loss

    @staticmethod
    @nnx.jit
    def eval_step(model, source_batch, target_batch):
        loss_fn = AutoencoderRegressor.loss_fn
        loss = loss_fn(model, source_batch, target_batch)
        return loss

    def get_bottleneck_and_mechanism_fcts(self):
        @nnx.jit
        def inference_step(model, source_batch):
            out_batch = model.encode(source_batch)
            return out_batch

        def fct(x):
            inference_dataloader = DataLoader(CBMDataset(x),
                                              batch_size=10000,
                                              shuffle=False,
                                              collate_fn=numpy_collate)
            z_out_list = []
            for batch in inference_dataloader:
                z_out_batch = inference_step(self.best_model, batch)
                z_out_list.append(z_out_batch)

            z_out = jnp.concatenate(z_out_list)
            return z_out

        return fct
