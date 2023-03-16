"""
 ~ CML // Creative Machine Learning ~
 tasks/classification.py : Simple helper functions for Panel styling
 
 For the interactive GUI aspects, we rely on the Panel library
 
 Author               :  Philippe Esling
                        <esling@ircam.fr>
"""

import jax.numpy as jnp
import inspect
import numpy as np
import param
import panel as pn
import distrax
from cml.randomness import Random
from cml.tasks.task import MLTask
from cml.plot import density, density_2d
from cml.data import gaussian


class DensityTask(MLTask):
    """Base class for subsquent density estimation tasks

    This class defines the overall signature of ML tasks represented as a 
    Panel GUI with interactive control. We need to define
        - generate_data:    How to generate the corresponding data
        - solve:            A solving method to approximate the problem
        - plot:             A plotting function to visualize the results
        - render:           Overall rendering of the Panel GUI

    Attributes (inherited):
        number_observations (int): Number of generated (observed) datapoints.
        noise_level (float): The noise level in the observations.
        ground_truth (bool): Display the ground truth function
        error (bool): Display the ground truth function
        data_code (bool): Display the data generation code
        solve_code (bool): Display the solver code
        title (str): Title of the figure
        equation (str): Corresponding equation
    """
    plot_type = param.Selector(default = "full")
    
    @param.depends('number_observations', 'noise_level')
    def plot(self):
        # Generate the data
        self.generate_data()
        # Retrieve the data
        x, x_p, p = self.x, self.x_p, self.p
        # Generate scatter
        if (self._dimension == 1):
            p = density(
                np.array(self.x_p),
                np.array(self.p),
                np.array(self.x),
                title = self.title,
                x_label = r"$$\color{white} x$$",
                y_label = r"$$\color{white} " + self.equation + "$$")
        else:
            p = density_2d(
                np.array(self.x_p),
                np.array(self.p),
                np.array(self.x),
                title = self.title,
                x_label = r"$$\color{white} x$$",
                y_label = r"$$\color{white} " + self.equation + "$$")
        return p
        
    def render(self):
        py_code = inspect.getsource(self.generate_data)[:-1]
        n_lines = py_code.count('\n')
        return pn.Column(
            pn.Row(
                pn.Param(self.param, name="Problem parameters"),
                self.plot),
            pn.widgets.Ace(value=py_code, sizing_mode='stretch_both', language='python', theme="chaos", height=30 * n_lines),
        )
    
class DensityGaussian(DensityTask):
    _dimension: int = 1
    mu = param.Number(default = 0.0, bounds=(-5.0, 5.0))
    sigma = param.Number(default = 1.0, bounds=(0.0, 5.0))

    def generate_data(self):
        self.distribution = distrax.Normal(self.mu, self.sigma)
        self.x = self.distribution.sample(seed=Random().split(), sample_shape=self.number_observations)
        self.x_p = jnp.linspace(self.mu - 4 * self.sigma, self.mu + 4 * self.sigma, 100)
        self.p = jnp.exp(self.distribution.log_prob(self.x_p))
    
    @param.depends('number_observations', 'noise_level', 'ground_truth', 'error', "mu", "sigma")
    def plot(self):
        # Set current labels
        self.title = r"Gaussian density"
        self.equation = r"p(\mathbf{x}) \sim \mathcal{N}" + f"({self.mu:.1f}, {self.sigma:.1f})"
        # Generate figure from super
        p = super().plot()
        return p
    
class DensityGaussian2D(DensityTask):
    _dimension: int = 2
    mu = param.Number(default = 0.0, bounds=(-5.0, 5.0))
    sigma = param.Number(default = 1.0, bounds=(0.0, 5.0))

    def generate_data(self):
        self.distribution = distrax.MultivariateNormal(self.mu, self.sigma)
        self.x = self.distribution.sample(seed=Random().split(), sample_shape=self.number_observations)
        self.x_p = jnp.linspace(self.mu - 4 * self.sigma, self.mu + 4 * self.sigma, 100)
        self.p = jnp.exp(self.distribution.log_prob(self.x_p))
    
    @param.depends('number_observations', 'noise_level', 'ground_truth', 'error', "mu", "sigma")
    def plot(self):
        # Set current labels
        self.title = r"Gaussian density"
        self.equation = r"p(\mathbf{x}) \sim \mathcal{N}" + f"({self.mu:.1f}, {self.sigma:.1f})"
        # Generate figure from super
        p = super().plot()
        return p
        
        
        
        
    