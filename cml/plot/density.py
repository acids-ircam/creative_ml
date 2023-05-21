"""
 ~ CML // Creative Machine Learning ~
 plot/density.py : Density plotting operations
 
 Author               :  Philippe Esling
                        <esling@ircam.fr>
"""

import torch
from typing import Callable
import jax.numpy as jnp
import numpy as np
from cml.plot import cml_figure, cml_figure_legend
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import torch.distributions as distribution
from matplotlib.patches import Ellipse

def density(
        x: jnp.ndarray, 
        density: jnp.ndarray,
        obs: jnp.ndarray,
        title: str = r"Observations",
        toolbar_location:str = None,
        x_label:str = "x",
        y_label:str = "y"):
    # Create a figure in our style
    p = cml_figure(
        plot_width=600, 
        plot_height=450, 
        toolbar_location=toolbar_location, 
        title=title)
    bins = np.linspace(-3, 3, 40)
    hist, edges = np.histogram(obs, density=True, bins=bins)
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
         fill_color="midnightblue", line_color="skyblue",
         legend_label="Samples")
    p.line(
        np.array(x), 
        np.array(density), 
        line_width=4, 
        line_alpha=0.7, 
        color="red",
        hover_alpha=0.5,
        hover_line_color="white",
        legend_label="Density")
    p.varea(np.array(x), np.zeros(density.shape), np.array(density), alpha=0.3, color="red")
    # Annotate labels
    p.xaxis.axis_label = x_label
    p.yaxis.axis_label = y_label
    # Add the legend
    p = cml_figure_legend(p)
    return p

def gaussian_ellipsoid(m, C, sdwidth=1, npts=None, axh=None, color='r'):
    # PLOT_GAUSSIAN_ELLIPSOIDS plots 2-d and 3-d Gaussian distributions
    if axh is None:
        axh = plt.gca()
    if m.size != len(m): 
        raise Exception('M must be a vector'); 
    if (m.size == 2):
        h = show2d(m[:], C, sdwidth, npts, axh, color)
    elif (m.size == 3):
        h = show3d(m[:], C, sdwidth, npts, axh, color)
    else:
        raise Exception('Unsupported dimensionality');
    return h

#-----------------------------
def show2d(means, C, sdwidth, npts=None, axh=None, color='r'):
    if (npts is None):
        npts = 50
    # plot the gaussian fits
    tt = np.linspace(0, 2 * np.pi, npts).transpose()
    x = np.cos(tt);
    y = np.sin(tt);
    ap = np.vstack((x[:], y[:])).transpose()
    v, d = np.linalg.eigvals(C)
    d = sdwidth / np.sqrt(d) # convert variance to sdwidth*sd
    bp = np.dot(v, np.dot(d, ap)) + means
    h = axh.plot(bp[:, 0], bp[:, 1], ls='-', color=color)
    return h

def show3d(means, C, sdwidth, npts=None, axh=None, color='r'):
    pass

def density_2d(
    prob_fn: Callable,
    n_points: int = 60,
    span: list[int] = [-4, 4]
    ):
    # Points
    N = n_points
    X = jnp.linspace(span[0], span[1], N)
    Y = jnp.linspace(span[0], span[1], N)
    X, Y = jnp.meshgrid(X, Y)
    # Pack X and Y into a single 3-dimensional array
    pos = jnp.zeros(X.shape + (2,))
    pos = pos.at[:, :, 0].set(X)
    pos = pos.at[:, :, 1].set(Y)
    # Evaluate density at points
    density = prob_fn(pos.reshape(-1, 2)).reshape(N, N)
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(18, 12), width_ratios=[.8, .2], height_ratios=[.2, .8])
    #ax = fig.gca()#projection='3d')
    # Plot level lines
    c_1 = axs[1, 0].contourf(X, Y, density, cmap=plt.cm.viridis)
    cs2 = axs[1, 0].contour(c_1, levels=c_1.levels[::1], colors='#EE6666')
    zc = cs2.collections[1]
    plt.setp(zc, linewidth=2, linestyle='--')
    # Plot contour
    cset = axs[1, 0].contourf(X, Y, density, cmap=plt.cm.viridis)
    plt.xlim([-3, 3]); plt.ylim([-3, 3])
    # Plot the marginal density
    axs[0, 0].plot(X[0, :], np.sum(density, axis=0), color='#EE6666')
    axs[0, 0].fill_between(X[0, :], np.sum(density, axis=0), 0, alpha=0.5, color='#EE6666')
    plt.xlim([-3, 3]); plt.ylim([-3, 3])
    # Plot the marginal density
    axs[1, 1].plot(np.sum(density, axis=1), Y[:, 0], color='#EE6666')
    axs[1, 1].fill_between(np.sum(density, axis=1), Y[:, 0], alpha=0.5, color='#EE6666')
    plt.xlim([-3, 3]); plt.ylim([-3, 3])
    plt.show()
    
def fit_multivariate_gaussian(X_s):
    gmm = GaussianMixture(n_components=1).fit(X_s)
    labels = gmm.predict(X_s)
    N = 50
    X = np.linspace(-2, 10, N)
    Y = np.linspace(-4, 4, N)
    X, Y = np.meshgrid(X, Y)
    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    norm = distribution.MultivariateNormal(torch.Tensor(gmm.means_[0]), torch.Tensor(gmm.covariances_[0]))
    Z = torch.exp(norm.log_prob(torch.Tensor(pos))).numpy()
    plt.figure(figsize=(10, 8));
    ax = plt.gca()
    ax.contourf(X, Y, Z, cmap='magma')
    plt.scatter(X_s[:, 0], X_s[:, 1], c='b', s=60, edgecolor='w', zorder=2.5); plt.grid(True);
    return labels

def fit_gaussian_mixture(X_s):
    gmm = GaussianMixture(n_components=4).fit(X_s)
    labels = gmm.predict(X_s)
    N = 50
    X = np.linspace(-2, 10, N)
    Y = np.linspace(-4, 4, N)
    X, Y = np.meshgrid(X, Y)
    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    Z = np.zeros((pos.shape[0], pos.shape[1]))
    for i in range(4):
        norm = distribution.MultivariateNormal(torch.Tensor(gmm.means_[i]), torch.Tensor(gmm.covariances_[i]))
        Z += torch.exp(norm.log_prob(torch.Tensor(pos))).numpy()
    plt.figure(figsize=(10, 8));
    ax = plt.gca()
    ax.contourf(X, Y, Z, cmap='magma')
    plt.scatter(X_s[:, 0], X_s[:, 1], c='b', s=60, edgecolor='w', zorder=2.5); plt.grid(True);
    return labels

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
def plot_gmm(gmm, X, label=True, ax=None):
    plt.figure(figsize=(10,8))
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='magma', edgecolor='gray', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.4 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)