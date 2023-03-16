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
from abc import abstractmethod
from cml.randomness import Random
from cml.tasks.task import MLTask
from cml.plot import scatter_classes
from cml.data import (
    linear_separation,
    xor_separation,
    two_circles,
    two_moons,
    make_blobs
)


class ClassificationTask(MLTask):
    """Base class for subsquent regression tasks

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
    _title: str = None
    _equation: str = None
    
    def generate_data(self):
        epsilon = Random().uniform(shape=self.x.shape, minval=-self.noise_level, maxval=self.noise_level)
        self.x_obs = self.x + epsilon
    
    #@param.depends('number_observations', 'noise_level')
    def plot(self):
        # Generate the data
        self.generate_data()
        # Retrieve the data
        x, y = self.x_obs, self.y
        # Generate scatter
        p = scatter_classes(
                np.array(x),
                np.array(y),
                title = self._title,
                x_label = r"$$\color{white} x$$",
                y_label = r"$$\color{white} " + self._equation + "$$")
        # Add the ground truth on top
        if (self.ground_truth):
            x, y = self.generate_ground_truth()
            p.line(np.array(x), np.array(y), line_width=8, line_alpha=0.7, color="red")
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
    
    @abstractmethod
    def plot_ground_truth(self):
        raise NotImplementedError("[ClassificationTask] - Abstract ground truth method")
    
    
class ClassificationLinear(ClassificationTask):
    """Linear classification problem

    Given a set of point uniformly distributed, we draw a random line following
        y = wx + b
    Creating a random linearly-separated classification problem
    
    """
    _title: str = r"Linear classification"
    _equation: str = r"\mathbf{W}\mathbf{x} + \mathbf{b}"
    data_generator = linear_separation

    def generate_data(self):
        self.x, self.y, (self.w, self.b) = linear_separation(self.number_observations)
        super().generate_data()
    
    def generate_ground_truth(self):
        x = np.linspace(np.min(self.x_obs[:, 0]), np.max(self.x_obs[:, 0]), 10)
        y = self.w * x + self.b
        return x, y
    
        
class ClassificationXOR(ClassificationTask):
    """XOR classification problem

    Draw two blobs representing each class based on randomly selected centers.
    
    """
    dense_factor = param.Number(default = 0.1, bounds=(0.01, 0.9))
    _title: str = r"XOR classification"
    _equation: str = r"\mathbf{y}"
    data_generator = xor_separation

    def generate_data(self):
        self.x, self.y = xor_separation(self.number_observations, self.dense_factor)
        super().generate_data()
    
    def generate_ground_truth(self):
        g_ref = np.linspace(0, 2 * np.pi, 10)
        x, y = jnp.cos(g_ref) * (self.factor * 1.1), jnp.sin(g_ref) * (self.factor * 1.1)
        return x, y
    
    
class ClassificationTwoCircles(ClassificationTask):
    """Two circles classification problem

    Draw two circles representing each class based on a given floating 
    factor of separation.
    
    """
    factor = param.Number(default = 0.1, bounds=(0.01, 0.9))
    _title: str = r"Two circles classification"
    _equation: str = r"\mathbf{y}"
    data_generator = two_circles

    def generate_data(self):
        self.x, self.y, (self.factor, ) = two_circles(self.number_observations, self.factor)
        super().generate_data()
    
    def generate_ground_truth(self):
        g_ref = np.linspace(0, 2 * np.pi, 10)
        x, y = jnp.cos(g_ref) * (self.factor * 1.1), jnp.sin(g_ref) * (self.factor * 1.1)
        return x, y
    
    
class ClassificationTwoMoons(ClassificationTask):
    """Two moons classification problem

    Draw two circles representing each class based on a given floating 
    factor of separation.
    
    """
    _title: str = r"Two moons classification"
    _equation: str = r"\mathbf{y}"
    data_generator = two_moons

    def generate_data(self):
        self.x, self.y, (self.factor, ) = two_moons(self.number_observations)
        super().generate_data()
    
    def generate_ground_truth(self):
        g_ref = np.linspace(0, 2 * np.pi, 10)
        x, y = jnp.cos(g_ref) * (self.factor * 1.1), jnp.sin(g_ref) * (self.factor * 1.1)
        return x, y
    
        
class ClassificationBlobs(ClassificationTask):
    """Two blobs classification problem

    Draw two blobs representing each class based on randomly selected centers.
    
    """
    _title: str = r"Blobs classification"
    _equation: str = r"\mathbf{y}"
    data_generator = make_blobs

    def generate_data(self):
        self.x, self.y, (self.centers, ) = make_blobs(self.number_observations)
        super().generate_data()
    
    def generate_ground_truth(self):
        g_ref = np.linspace(0, 2 * np.pi, 10)
        x, y = jnp.cos(g_ref) * (self.factor * 1.1), jnp.sin(g_ref) * (self.factor * 1.1)
        return x, y
    
        
        
        
        
    