"""
 ~ CML // Creative Machine Learning ~
 __init__.py : Initializing 
 
 For the interactive GUI aspects, we rely on the Panel library
 
 Author               :  Philippe Esling
                        <esling@ircam.fr>
"""
import os

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

# Data generation
from cml.data import (
    polynomial,
    swiss_roll
)
# Randomness generator
from cml.randomness import (
    Random
)
# Tasks GUI objects
from cml.tasks import (
    MLTask,
    RegressionPolynomial
)
# Panel-related methods
from cml.panel import (
    initialize_panel
)
# Plot-related methods
from cml.plot import (
    initialize_bokeh,
    cml_figure
)

def initialize_cml_environment():
    initialize_bokeh()
    initialize_panel()

__all__ = [
    "polynomial",
    "swiss_roll",
    "Random",
    "MLTask",
    "RegressionPolynomial",
    "initialize_bokeh",
    "initialize_cml_environment"
    "cml_figure"
]
