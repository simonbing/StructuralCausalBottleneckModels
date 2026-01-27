import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx


class Autoencoder(nnx.Module):
    """
    Simple autoencoder model.
    """
    def __init__(self, in_dim, dense_x_z, dense_z_x, z_dim, cond_dim, out_dim, rngs):
        key = rngs.params()
        self.rngs = rngs

        # Encoder params
        layers_MLP = []
        for i in range(len(dense_x_z)):
            if i == 0:
                layers_MLP.append(nnx.Linear(in_dim, dense_x_z[i], rngs=rngs))
            else:
                layers_MLP.append(nnx.Linear(dense_x_z[i-1], dense_x_z[i], rngs=rngs))
            layers_MLP.append(nnx.swish) 
        self.encoder_final = nnx.Linear(dense_x_z[-1], z_dim, rngs=rngs)
        self.encoder = nnx.Sequential(*layers_MLP)

        # Decoder params
        z_cond_dim = z_dim + cond_dim
        layers_MLP = []
        for i in range(len(dense_z_x)):
            if i == 0:
                layers_MLP.append(nnx.Linear(z_cond_dim, dense_z_x[i], rngs=rngs))
            else:
                layers_MLP.append(nnx.Linear(dense_z_x[i-1], dense_z_x[i], rngs=rngs))
            layers_MLP.append(nnx.swish)
        layers_MLP.append(nnx.Linear(dense_z_x[-1], out_dim, rngs=rngs))
        self.decoder = nnx.Sequential(*layers_MLP)

    def encode(self, x):
        h = self.encoder(x)
        z = self.encoder_final(h)
        return z

    def decode(self, z):
        return self.decoder(z)

    def __call__(self, *x):
        """
        x is expected to be a list, either with a single entry (if there is no
        conditioning), or with two entries: [X, X_cond].
        """
        # TODO: get X and X_cond here if necessary
        x_in = x[0]

        z = self.encode(x_in)

        if len(x) > 1:  # if there is a conditioning variable, concat it here
            x_cond = x[1]
            z_cat = jnp.concatenate((z, x_cond), axis=1)
        else:
            z_cat = z

        x_out = self.decode(z_cat)

        return x_out
    

class VAE(Autoencoder):
    """
    Simple variational autoencoder model.
    """
    def __init__(self, in_dim, dense_x_z, dense_z_x, z_dim, cond_dim, out_dim, rngs):
        super().__init__(in_dim, dense_x_z, dense_z_x, z_dim, cond_dim, out_dim, rngs)

        # Additional layers for mean and logvar
        self.fc_mu = nnx.Linear(dense_x_z[-1], z_dim, rngs=rngs)
        self.fc_logvar = nnx.Linear(dense_x_z[-1], z_dim, rngs=rngs)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = jnp.exp(0.5 * logvar)

        eps = jax.random.normal(self.rngs.reparam(), std.shape)
        
        return mu + eps * std

    def __call__(self, *x):
        x_in = x[0]

        mu, logvar = self.encode(x_in)
        z = self.reparameterize(mu, logvar)

        if len(x) > 1:  # if there is a conditioning variable, concat it here
            x_cond = x[1]
            z_cat = jnp.concatenate((z, x_cond), axis=1)
        else:
            z_cat = z

        x_out = self.decode(z_cat)

        return x_out, mu, logvar
