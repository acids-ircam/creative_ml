"""
 ~ CML // Creative Machine Learning ~
 tasks.py : Simple helper functions for Panel styling
 
 For the interactive GUI aspects, we rely on the Panel library
 
 Author               :  Philippe Esling
                        <esling@ircam.fr>
"""

import jax.numpy as jnp
import inspect
import numpy as np
import param
import panel as pn
from typing import Callable
from cml.randomness import Random
from cml.tasks.task import MLTask
from cml.plot import scatter
from cml.data import polynomial, linear, swiss_roll
from cml.utils import remove_comments


class RegressionTask(MLTask):
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
        self.y_obs = self.y + epsilon
    
    def plot(self):
        # Generate the data
        self.generate_data()
        # Retrieve the data
        x, y = self.x, self.y_obs
        # Generate scatter
        p = scatter(
                np.array(x),
                np.array(y),
                title = self._title,
                x_label = r"$$\color{white} x$$",
                y_label = r"$$\color{white} " + self._equation + "$$")
        # Add true function
        if (self.ground_truth):
            p.line(np.array(self.x), np.array(self.y), line_width=4, line_alpha=0.6, legend_label=r"True function")
        # Add error bars
        if (self.error):
            x = np.array(self.x)
            y = np.array(self.y_obs)
            poly = np.array(self.y)
            for i in range(len(x)):
                p.line([x[i], x[i]], [poly[i], y[i]], color="red")
        return p
        
    def render(self):
        py_code = remove_comments(inspect.getsource(self.data_generator))[:-1]
        n_lines = py_code.count('\n')
        return pn.Row(
            pn.layout.HSpacer(),
            pn.Column(
                pn.Row(
                    pn.Param(self.param, 
                             name="Problem parameters",
                             widgets = self.widgets),
                    self.plot),
                pn.widgets.CodeEditor(value=py_code, language='python', theme="chaos", height=30 * n_lines, width=640),  
            ),
            pn.layout.HSpacer(),
            )

class RegressionLinear(RegressionTask):
    """Regression task on a linear problem

    Attributes:
        x_1 (float): 1st order coefficient
        x_0 (float): 0th order coefficient
    """
    x_1 = param.Number(default = 1.0, bounds=(-5.0, 5.0))
    x_0 = param.Number(default = 0.0, bounds=(-5.0, 5.0))
    _title: str = r"Linear regression problem"
    data_generator = linear

    def generate_data(self):
        # Generating polynomial regression problem
        self.x, self.y = linear(jnp.array([self.x_1, self.x_0]), self.number_observations)
        super().generate_data()
    
    def plot(self):
        # Set current label dynamically
        self._equation = f"{self.x_1:.1f}x^{1} + {self.x_0:.1f}"
        # Generate figure from super
        return super().plot()

class RegressionPolynomial(RegressionTask):
    """Regression task on a degree 2 polynomial.

    Attributes:
        x_2 (float): 2nd order coefficient
        x_1 (float): 1st order coefficient
        x_0 (float): 0th order coefficient
    """
    x_2 = param.Number(default = -2.0, bounds=(-5.0, 5.0))
    x_1 = param.Number(default = 0.0, bounds=(-5.0, 5.0))
    x_0 = param.Number(default = 1.0, bounds=(-5.0, 5.0))
    _title: str = r"Polynomial regression problem"
    data_generator = polynomial

    def generate_data(self):
        # Generating polynomial regression problem
        self.x, self.y = polynomial(jnp.array([self.x_2, self.x_1, self.x_0]), self.number_observations)
        super().generate_data()
    
    def plot(self):
        # Set current label dynamically
        self._equation = f"{self.x_2:.1f}x^{2} + {self.x_1:.1f}x^{1} + {self.x_0:.1f}x^{0}"
        # Generate figure from super
        return super().plot()
    
    
class RegressionSwissRoll(RegressionTask):
    _title: str = r"Swiss roll subsampling"
    data_generator = swiss_roll

    def generate_data(self):
        # Generating learning problem
        self.x, self.y = swiss_roll(self.number_observations)
        super().generate_data()
    
    def plot(self):
        # Set current label dynamically
        self._equation = f"\epsilon = {self.noise_level}"
        # Generate figure from super
        return super().plot()

class RegressionLinearSolver(RegressionLinear):
    
    def plot(self):
        p = super().plot()
        x_predict, y_model = self.solve(self.x, self.y_obs)
        p.line(np.array(x_predict), y_model, line_width=4, line_alpha=0.7, color="red", legend_label=r"Learned model")
        return p

class RegressionPolynomialSolver(RegressionPolynomial):
    degree = param.Integer(default=1, bounds=(1, 30))
    
    def plot(self):
        p = super().plot()
        x_predict, y_model = self.solve(self.x, self.y_obs, self.degree)
        p.line(np.array(x_predict), y_model, line_width=4, line_alpha=0.7, color="red")
        return p
    
    
