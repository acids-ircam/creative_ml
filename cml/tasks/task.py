"""
 ~ CML // Creative Machine Learning ~
 tasks.py : Simple helper functions for Panel styling
 
 For the interactive GUI aspects, we rely on the Panel library
 
 Author               :  Philippe Esling
                        <esling@ircam.fr>
"""

import param
import panel as pn
from abc import abstractmethod
from typing import Callable


class MLTask(param.Parameterized):
    """Base class for all subsequent ML tasks.

    This class defines the overall signature of ML tasks represented as a 
    Panel GUI with interactive control. We need to define
        - generate_data:    How to generate the corresponding data
        - solve:            A solving method to approximate the problem
        - plot:             A plotting function to visualize the results
        - render:           Overall rendering of the Panel GUI

    Attributes:
        number_observations (int): Number of generated (observed) datapoints.
        noise_level (float): The noise level in the observations.
        ground_truth (bool): Display the ground truth function
        error (bool): Display the ground truth function
        data_code (bool): Display the data generation code
        solve_code (bool): Display the solver code
        title (str): Title of the figure
        equation (str): Corresponding equation

    """
    number_observations = param.Integer(default=100, bounds=(10, 1000))
    noise_level = param.Number(default = 0.1, bounds=(0.0, 5.0))
    ground_truth = param.Boolean(default=False)
    error = param.Boolean(default=False)
    data_code: bool = False
    solve_code: bool = False
    title: str = None
    equation: str = None
    data_generator: Callable = None
    widgets: dict = {}
    
    @abstractmethod
    def generate_data(self):
        """
        Data generation function, it should fill the corresponding class 
        attributes depending on the task at hand.
        """
        raise NotImplementedError("[MLTask] - Cannot generate data for abstract task")
    
    @abstractmethod
    def solve(self):
        """
        Solve the corresponding approximation task.
        """
        raise NotImplementedError("[MLTask] - Cannot plot data for abstract task")
        
    @abstractmethod
    def plot(self):
        """
        Plot the task, corresponding function and approximation.
        """
        raise NotImplementedError("[MLTask] - Cannot plot data for abstract task")
        
    def render(self):
        """
        Perform GUI rendering with Panel
        """
        return pn.Row(
            pn.Param(self.param, name="Problem parameters"),
            self.plot)