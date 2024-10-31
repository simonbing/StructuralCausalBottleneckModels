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

        # Encoder params
        layers_MLP = []
        for i in range(len(dense_x_z)):
            if i == 0:
                layers_MLP.append(nnx.Linear(in_dim, dense_x_z[i], rngs=rngs))
            else:
                layers_MLP.append(nnx.Linear(dense_x_z[i-1], dense_x_z[i], rngs=rngs))
            layers_MLP.append(nnx.swish)  # TODO: check if this works or we need a different nonlinearity!
            # layers_MLP.append(nnx.leaky_relu)
            # Not using any dropout for this simple model
        layers_MLP.append(nnx.Linear(dense_x_z[-1], z_dim, rngs=rngs))
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
            # layers_MLP.append(nnx.leaky_relu)
        layers_MLP.append(nnx.Linear(dense_z_x[-1], out_dim, rngs=rngs))
        self.decoder = nnx.Sequential(*layers_MLP)

    def encode(self, x):
        return self.encoder(x)

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
