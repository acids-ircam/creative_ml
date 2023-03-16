"""
 ~ CML // Creative Machine Learning ~
 data/density.py : Simple functions for generating density estimation tasks
 
 Currently implemented toy data generation are
     - Normal
     - Bivariate 
     - Ring 
     - Wave 
     - Wave twist
     - Wave split
     - Circle (of Gaussians)
     - Grid (of Gauss)
 
 Author               :  Philippe Esling
                        <esling@ircam.fr>
"""

import jax.numpy as jnp
import numpy as np
import math

"""
 ~~~~~~~~~~~~~~~~~~~~
 
 Density estimation tasks
 
 ~~~~~~~~~~~~~~~~~~~~
"""

w1 = lambda z: jnp.sin(2 * jnp.pi * z[:, 0] / 4)
w2 = lambda z: 3 * jnp.exp(-0.5 * ((z[:, 0] - 1) / 0.6) ** 2)
w3 = lambda z: 3 * (1.0 / (1 + jnp.exp(-(z[:, 0] - 1) / 0.3)))

def gaussian(z, 
             mu=jnp.zeros((1, 2)), 
             sig=jnp.ones((1, 2))):
    """
    

    Parameters
    ----------
    z : TYPE
        DESCRIPTION.
    mu : TYPE, optional
        DESCRIPTION. The default is jnp.zeros((1, 2)).
    sig : TYPE, optional
        DESCRIPTION. The default is jnp.ones((1, 2)).

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    z = z[:, None, :] - mu[None, :, :]
    sig_inv = 1./sig
    exponent = -0.5 * jnp.sum(z * sig_inv[None, :, :] * z, (1, 2))
    return jnp.exp(exponent)

def bivariate(z):
    add1 = 0.5 * ((jnp.linalg.norm(z, 2, 1) - 2) / 0.4) ** 2
    add2 = -jnp.log(jnp.exp(-0.5 * ((z[:, 0] - 2) / 0.6) ** 2) + jnp.exp(-0.5 * ((z[:, 0] + 2) / 0.6) ** 2))
    return jnp.exp(-(add1 + add2))


def ring(z):
    z1, z2 = jnp.split(z, 2, axis=1)
    norm = jnp.sqrt(z1 ** 2 + z2 ** 2)
    exp1 = jnp.exp(-0.5 * ((z1 - 2) / 0.8) ** 2)
    exp2 = jnp.exp(-0.5 * ((z1 + 2) / 0.8) ** 2)
    u = 0.5 * ((norm - 4) / 0.4) ** 2 - jnp.log(exp1 + exp2)
    return jnp.exp(-u)


def wave(z):
    z = jnp.reshape(z, [z.shape[0], 2])
    z1, z2 = z[:, 0], z[:, 1]
    u = 0.5 * ((z2 - w1(z))/0.4) ** 2
    u = u.at[jnp.abs(z1) > 4].set(1e8)
    return jnp.exp(-u)


def wave_twist(z):
    in1 = jnp.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.35) ** 2)
    in2 = jnp.exp(-0.5 * ((z[:, 1] - w1(z) + w2(z)) / 0.35) ** 2)
    return jnp.exp(jnp.log(in1 + in2 + 1e-9))


def wave_split(z):
    in1 = jnp.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2)
    in2 = jnp.exp(-0.5 * ((z[:, 1] - w1(z) + w3(z)) / 0.35) ** 2)
    return jnp.exp(jnp.log(in1 + in2))


def circle(z, n_dens=10):
    full_dens = []
    for n in range(n_dens):
        x = math.cos((n / float(n_dens)) * 2 * math.pi) * 3
        y = math.sin((n / float(n_dens)) * 2 * math.pi) * 3
        cur_g = gaussian(z, mu=jnp.array([[x, y]]), sig= 0.1 * jnp.ones((1, 2)))
        full_dens.append(jnp.expand_dims(cur_g, axis = 1))
    return jnp.sum(jnp.concatenate(full_dens, axis=1), axis=1)


def grid(z, 
         n_rows: int = 4, 
         n_cols: int = 4,
         span: list[int] = [-3, 3],
         var: float = 0.02):
    full_dens = []
    pos_rows = np.linspace(span[0], span[1], n_rows)
    pos_cols = np.linspace(span[0], span[1], n_cols)
    for i in range(n_rows):
        for j in range(n_cols):
            x, y = pos_cols[i], pos_rows[j]
            cur_g = gaussian(z, mu = jnp.array([[x, y]]), sig = var * jnp.ones((1, 2)))
            full_dens.append(jnp.expand_dims(cur_g, axis = 1))
    return jnp.sum(jnp.concatenate(full_dens, axis=1), axis=1)


def get_density(args):
    arg = args if isinstance(args, str) else args.density
    densities = {
            'bivar':bivariate, 
            'ring':ring, 
            'wave':wave, 
            'wave_twist':wave_twist, 
            'wave_split':wave_split, 
            'circle':circle,
            'grid':grid
            }
    return densities[arg]