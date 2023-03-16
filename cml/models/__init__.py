"""
 ~ CML // Creative Machine Learning ~
 models/__init__.py : Initializing models-related methods
 
 Author               :  Philippe Esling
                        <esling@ircam.fr>
"""
from bokeh.io import output_notebook
from bokeh.plotting import curdoc


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

__all__ = [
    "initialize_bokeh",
    "cml_figure",
    "cml_figure_legend",
    "cml_figure_colorbar",
    "cml_figure_axis",
    "scatter"
]
