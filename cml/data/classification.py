"""
 ~ CML // Creative Machine Learning ~
 data/regression.py : Simple functions for generating regression tasks
 
 Currently implemented toy data generation are
    - Polynomial
    - Swiss roll
 
 Author               :  Philippe Esling
                        <esling@ircam.fr>
"""

import math
import jax.numpy as jnp
from cml.randomness import Random

def classification(
        n_observations: int = 100
        ):
    """Generic description of classification data generation.    

    Parameters
    ----------
    n_observations : int, optional
        Number of sampling observations to generate. The default is 100.

    """
    pass


def linear_separation(
        n_observations: int = 100
        ):
    """Linearly separated classification problem.

    Parameters
    ----------
    n_observations : int, optional
        Number of sampling observations to generate. The default is 100.

    """
    # Generate 2-dimensional random points
    x = Random().uniform(shape = (int(n_observations), 2)) * 2 - 1;
    # Slope of separating line
    slope = jnp.log(Random().uniform() * 10);
    yint = Random().uniform() * 2 - 1;
    # Create the indexes for a two-class problem
    y = (x[:, 0] - x[:, 1] * slope - yint > 0) * 1;
    # Plot the corresponding pattern
    return x, y, (1.0 / slope, - (yint / slope))


def xor_separation(
        n_observations: int = 100,
        dense_factor: float = 0.01
        ):
    """XOR classification problem.

    Parameters
    ----------
    n_observations : int, optional
        Number of sampling observations to generate. The default is 100.

    """
    # Create base patterns
    x_b = jnp.array([[-1, -1],[-1,  1],[1, -1],[1,  1]])
    # Corresponding classes
    y_b = jnp.array([0, 1, 1, 0])
    x, y = [], []               
    # Generate 2-dimensional random points
    n_augment = (n_observations - 4) // 4
    for p in range(4):
       rep_pat = jnp.repeat(x_b[p, :][jnp.newaxis, :], n_augment, axis=0)
       x.append(Random().normal(shape=(n_augment, 2)) * dense_factor + rep_pat)
       y.append(jnp.repeat(y_b[p], n_augment))
    x = jnp.concatenate(x, axis=0)
    y = jnp.concatenate(y)
    # Plot the corresponding pattern
    return x, y


def two_circles(
        n_observations: int = 100, 
        factor: float = 0.8
        ):
    """Make a large circle containing a smaller circle in 2d.
    
    Parameters
    ----------
    n_observations : int or tuple of shape (2,), dtype=int, default=100
        If int, the total number of points generated.
        If two-element tuple, number of points in each class.
    factor : float, default=.8
        Scale factor between inner and outer circle in the range `(0, 1)`.
    Returns
    -------
    X : ndarray of shape (n_samples, 2)
        The generated samples.
    y : ndarray of shape (n_samples,)
        The integer labels (0 or 1) for class membership of each sample.
    """
    if (isinstance(n_observations, int)):
        n_in = n_observations // 2
        n_out = n_observations - n_in
    else:
        n_in, n_out = n_observations
    # Generate angular references
    a_in = jnp.linspace(0, 2 * jnp.pi, n_in)
    a_out = jnp.linspace(0, 2 * jnp.pi, n_out)
    in_x = jnp.cos(a_in) * factor
    in_y = jnp.sin(a_in) * factor
    out_x = jnp.cos(a_out)
    out_y = jnp.sin(a_out)
    # Create observations
    x = jnp.vstack([jnp.append(out_x, in_x), jnp.append(out_y, in_y)]).T
    y = jnp.hstack([jnp.zeros(n_out, dtype=jnp.int16), jnp.ones(n_in, dtype=jnp.int16)])
    return x, y, (factor, )


def two_moons(
        n_observations: int = 100
        ):
    """Make two interleaving half circles.

    A simple toy dataset to visualize clustering and classification
    algorithms. Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_observations : int or tuple of shape (2,), dtype=int, default=100
        If int, the total number of points generated.
        If two-element tuple, number of points in each class.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
        The generated samples.

    y : ndarray of shape (n_samples,)
        The integer labels (0 or 1) for class membership of each sample.
    """
    if (isinstance(n_observations, int)):
        n_in = n_observations // 2
        n_out = n_observations - n_in
    else:
        n_in, n_out = n_observations
    out_x = jnp.cos(jnp.linspace(0, jnp.pi, n_out))
    out_y = jnp.sin(jnp.linspace(0, jnp.pi, n_out))
    in_x = 1 - jnp.cos(jnp.linspace(0, jnp.pi, n_in))
    in_y = 1 - jnp.sin(jnp.linspace(0, jnp.pi, n_in)) - 0.5
    x = jnp.vstack([jnp.append(out_x, in_x), jnp.append(out_y, in_y)]).T
    y = jnp.hstack([jnp.zeros(n_out, dtype=jnp.int16), jnp.ones(n_in, dtype=jnp.int16)])
    return x, y, (0, )


def make_blobs(
        n_observations=100,
        n_features=2,
        centers=None,
        cluster_std=1.0,
        center_box=(-10.0, 10.0),
    ):
    """Generate isotropic Gaussian blobs for clustering.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int or array-like, default=100
        If int, it is the total number of points equally divided among
        clusters.
        If array-like, each element of the sequence indicates
        the number of samples per cluster.
    n_features : int, default=2
        The number of features for each sample.

    centers : int or ndarray of shape (n_centers, n_features), default=None
        The number of centers to generate, or the fixed center locations.
        If n_samples is an int and centers is None, 3 centers are generated.
        If n_samples is array-like, centers must be
        either None or an array of length equal to the length of n_samples.

    cluster_std : float or array-like of float, default=1.0
        The standard deviation of the clusters.

    center_box : tuple of float (min, max), default=(-10.0, 10.0)
        The bounding box for each cluster center when centers are
        generated at random.
        
    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The generated samples.

    y : ndarray of shape (n_samples,)
        The integer labels for cluster membership of each sample.

    centers : ndarray of shape (n_centers, n_features)
        The centers of each cluster. Only returned if
        ``return_centers=True``.

    """
    if centers is None:
        centers = 3
    if type(centers) == int:
        n_centers = centers
        centers = Random().uniform(minval=center_box[0], maxval=center_box[1], shape=(n_centers, n_features))
    else:
        n_features = centers.shape[1]
        n_centers = centers.shape[0]
    if type(cluster_std) == float:
        cluster_std = jnp.full(len(centers), cluster_std)
    n_obs_center = [int(n_observations // n_centers)] * n_centers
    for i in range(n_observations % n_centers):
        n_obs_center[i] += 1
    cum_sum_n_samples = jnp.cumsum(jnp.array(n_obs_center))
    x = jnp.empty(shape=(sum(n_obs_center), n_features), dtype=jnp.float32)
    y = jnp.empty(shape=(sum(n_obs_center),), dtype=jnp.int16)
    for i, (n, std) in enumerate(zip(n_obs_center, cluster_std)):
        start_idx = cum_sum_n_samples[i - 1] if i > 0 else 0
        end_idx = cum_sum_n_samples[i]
        x = x.at[start_idx:end_idx].set(Random().normal(shape=(n, n_features)) * std + centers[i])
        y = y.at[start_idx:end_idx].set(i)
    return x, y, (centers, )