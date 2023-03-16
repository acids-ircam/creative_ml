"""
 ~ CML // Creative Machine Learning ~
 plot/__init__.py : Initializing plot-related methods
 
 Author               :  Philippe Esling
                        <esling@ircam.fr>
"""
from bokeh.io import output_notebook
from bokeh.plotting import curdoc

def initialize_bokeh():
    curdoc().theme = "dark_minimal"
    output_notebook()

from cml.plot.style import (
    cml_figure,
    cml_figure_legend,
    cml_figure_colorbar,
    cml_figure_axis
)
from cml.plot.basic import (
    scatter,
    scatter_classes
)
from cml.plot.density import (
    density,
    density_2d
)

__all__ = [
    "initialize_bokeh",
    "cml_figure",
    "cml_figure_legend",
    "cml_figure_colorbar",
    "cml_figure_axis",
    "scatter",
    "scatter_classes",
    "density",
    "density_2d"
]
