"""
 ~ CML // Creative Machine Learning ~
 data/regression.py : Simple functions for generating regression tasks
 
 Currently implemented toy data generation are
    - Polynomial
    - Swiss roll
 
 Author               :  Philippe Esling
                        <esling@ircam.fr>
"""

#%% -*- coding: utf-8 -*-

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import math

"""
 ~~~~~~~~~~~~~~~~~~~~
 
 Regression tasks
 
 ~~~~~~~~~~~~~~~~~~~~
"""

def polynomial(
        coefficients: jnp.ndarray,
        n_observations: int = 100) -> [jnp.ndarray, jnp.ndarray]:
    """
    Generate a 1-dimensional polynomial of variable degrees given some 
    input coefficients

    Parameters
    ----------
    coefficients : jnp.ndarray
        Sets of polynomial coefficients ordered from the highest degree.
    n_observations : int, optional
        Number of observations to generate. The default is 100.

    Returns
    -------
    x : jnp.ndarray
        Set of X coordinates.
    y : jnp.ndarray
        Set of Y coordinates.

    """
    x = jnp.linspace(0, 1, n_observations)
    y = jnp.polyval(coefficients, x)
    return x, y


def linear(
        coefficients: jnp.ndarray,
        n_observations: int = 100) -> [jnp.ndarray, jnp.ndarray]:
    """
    Generate a 1-dimensional linear set given some input coefficients

    Parameters
    ----------
    coefficients : jnp.ndarray
        First coefficient and bias
    n_observations : int, optional
        Number of observations to generate. The default is 100.

    Returns
    -------
    x : jnp.ndarray
        Set of X coordinates.
    y : jnp.ndarray
        Set of Y coordinates.

    """
    x = jnp.linspace(0, 1, n_observations)
    y = coefficients[0] * x + coefficients[1]
    return x, y


def swiss_roll(
        n_observations: int = 100) -> [jnp.ndarray, jnp.ndarray]:
    """
    Generate a typical Swiss Roll dataset

    Parameters
    ----------
    n_observations : int, optional
        Number of observations to generate. The default is 100.

    Returns
    -------
    x : jnp.ndarray
        Set of X coordinates.
    y : jnp.ndarray
        Set of Y coordinates.

    """
    t = np.linspace(0,4 * np.pi, n_observations)
    r = np.linspace(0.1, 1, n_observations)
    x, y =  r * np.cos(t), r * np.sin(t)
    return x, y



def make_s_curve(n_samples=100, *, noise=0.0, random_state=None):
    """Generate an S curve dataset.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, default=100
        The number of sample points on the S curve.

    noise : float, default=0.0
        The standard deviation of the gaussian noise.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : ndarray of shape (n_samples, 3)
        The points.

    t : ndarray of shape (n_samples,)
        The univariate position of the sample according to the main dimension
        of the points in the manifold.
    """
    generator = check_random_state(random_state)

    t = 3 * np.pi * (generator.uniform(size=(1, n_samples)) - 0.5)
    X = np.empty(shape=(n_samples, 3), dtype=np.float64)
    X[:, 0] = np.sin(t)
    X[:, 1] = 2.0 * generator.uniform(size=n_samples)
    X[:, 2] = np.sign(t) * (np.cos(t) - 1)
    X += noise * generator.standard_normal(size=(3, n_samples)).T
    t = np.squeeze(t)

    return X, t