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
from cbm.estimation.jax_models import Autoencoder, VAE
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
                                 rngs=nnx.Rngs(params=int(self.seed)))

        # self.model = MLP(d=self.d_micro_in, dense_layers=[128, 128, 128, 128],
        #                  rngs=nnx.Rngs(params=self.seed))

        self.epochs = epochs
        self.batch_size = batch_size

        # lr_scheduler = optax.exponential_decay(init_value=learning_rate,
        #                                        transition_steps=100,
        #                                        decay_rate=0.9,
        #                                        staircase=True)
        
        lr_scheduler = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=1000,  # 600
            decay_steps=10000,  # 3000
            end_value=1e-7,
        )

        self.composed_optimizer = optax.chain(
            # optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=lr_scheduler, b1=momentum)
        )

        self.optimizer = nnx.Optimizer(self.model, self.composed_optimizer)

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
            epoch_train_loss = 0
            for batch in train_dataloader:
                source_batch = batch[:-1]  # also includes X_cond if it's there
                target_batch = batch[-1]
                loss = self.train_step(self.model, self.optimizer, source_batch, target_batch)
                epoch_train_loss += loss
                wandb.log({f'train loss ({self.source}, {self.target}), estim': loss,
                           f'step_train ({self.source}, {self.target}), estim': log_step_train})
                log_step_train += 1
            epoch_train_loss = epoch_train_loss / len(train_dataloader)
            wandb.log({f'epoch train loss ({self.source}, {self.target}), estim': epoch_train_loss,
                       f'epoch': epoch})
            # Eval
            epoch_eval_loss = 0
            for batch in val_dataloader:
                source_batch = batch[:-1]  # also includes X_cond if it's there
                target_batch = batch[-1]
                eval_loss = self.eval_step(self.model, source_batch, target_batch)
                ### DEBUG: take eval loss with best mode
                # Check if best_model is defined
                # if 'best_model' not in locals():
                #     eval_loss = self.eval_step(self.model, source_batch, target_batch)
                # else:
                #     eval_loss = self.eval_step(best_model, source_batch, target_batch)
                ###
                epoch_eval_loss += eval_loss
                wandb.log({f'eval loss ({self.source}, {self.target}), estim': eval_loss,
                           f'step_eval ({self.source}, {self.target}), estim': log_step_eval})
                log_step_eval += 1
            epoch_eval_loss = epoch_eval_loss / len(val_dataloader)
            wandb.log({f'epoch eval loss ({self.source}, {self.target}), estim': epoch_eval_loss,
                       f'epoch': epoch})
            if epoch_eval_loss < best_eval_loss:
                best_model = copy.deepcopy(self.model)
                best_eval_loss = eval_loss

        self.best_model = best_model

        ### DEBUG: Use last model
        # self.best_model = self.model
        ###

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

        # TODO: add mechanism function, currently returning none

        return fct, None
    

class VariationalAutoencoderRegressor(AutoencoderRegressor):
    def __init__(self, seed, beta, d_micro_in, d_micro_out, d_bottleneck, source,
                 target, d_cond, dense_x_z, dense_z_x, epochs, batch_size,
                 learning_rate, momentum):
        super().__init__(seed, d_micro_in, d_micro_out, d_bottleneck, source,
                         target, d_cond, dense_x_z, dense_z_x, epochs,
                         batch_size, learning_rate, momentum)
        self.__class__.beta = beta

        # Override model with VAE
        self.model = VAE(in_dim=self.d_micro_in, dense_x_z=dense_x_z,
                         dense_z_x=dense_z_x, z_dim=self.d_bottleneck,
                         cond_dim=self.d_cond, out_dim=self.d_micro_out,
                         rngs=nnx.Rngs(params=int(self.seed),
                                       reparam=int(self.seed)+1))  # Additional prng stream for reparametrization trick
        
        self.optimizer = nnx.Optimizer(self.model, self.composed_optimizer)
        
    @staticmethod
    def loss_fn(model, source_batch, target_batch):
        mu, logvar = model.encode(source_batch[0])
        z = model.reparameterize(mu, logvar)
        if len(source_batch) > 1:
            x_cond = source_batch[1]
            z_cat = jnp.concatenate((z, x_cond), axis=1)
        else:
            z_cat = z
        out_batch = model.decode(z_cat)

        # Reconstruction loss
        recon_loss = jnp.sum((out_batch - target_batch) ** 2, axis=1).mean()
        # KL divergence loss
        kl_loss = -0.5 * jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar), axis=1).mean()
        # Total loss
        loss = recon_loss + VariationalAutoencoderRegressor.beta * kl_loss
        return loss
    
    @staticmethod
    @nnx.jit
    def train_step(model, optimizer, source_batch, target_batch):
        grad_fn = nnx.value_and_grad(VariationalAutoencoderRegressor.loss_fn, has_aux=False)
        loss, grads = grad_fn(model, source_batch, target_batch)
        optimizer.update(grads)
        return loss

    @staticmethod
    @nnx.jit
    def eval_step(model, source_batch, target_batch):
        loss_fn = VariationalAutoencoderRegressor.loss_fn
        loss = loss_fn(model, source_batch, target_batch)
        return loss
    
    def get_bottleneck_and_mechanism_fcts(self):
        @nnx.jit
        def inference_step(model, source_batch):
            mu_batch, logvar_batch = model.encode(source_batch)
            # out_batch = model.reparameterize(mu_batch, logvar_batch)
            return mu_batch  # Return mean as bottleneck representation

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

        # TODO: add mechanism function, currently returning none

        return fct, None
