"""
 ~ CML // Creative Machine Learning ~
 plot/__init__.py : Initializing plot-related methods
 
 Author               :  Philippe Esling
                        <esling@ircam.fr>
"""
from bokeh.io import output_notebook
from bokeh.plotting import curdoc
import panel as pn

def center_plot(p):
    return pn.Row(pn.layout.HSpacer(), p, pn.layout.HSpacer())

def initialize_bokeh():
    curdoc().theme = "dark_minimal"
    output_notebook()

from cml.plot.style import (
    cml_figure,
    cml_figure_legend,
    cml_figure_colorbar,
    cml_figure_axis,
    cml_figure_matplotlib
)
from cml.plot.basic import (
    scatter,
    scatter_classes,
    scatter_boundary,
    image
)
from cml.plot.density import (
    density,
    density_2d,
    gaussian_ellipsoid
)

__all__ = [
    "center_plot"
    "initialize_bokeh",
    "cml_figure",
    "cml_figure_legend",
    "cml_figure_colorbar",
    "cml_figure_axis",
    "cml_figure_matplotlib"
    "scatter",
    "scatter_classes",
    "scatter_boundary",
    "image",
    "density",
    "density_2d",
    "gaussian_ellipsoid"
]
