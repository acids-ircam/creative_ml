"""
 ~ CML // Creative Machine Learning ~
 plot/basic.py : Basic plotting operations
 
 Author               :  Philippe Esling
                        <esling@ircam.fr>
"""

import numpy as np
import jax.numpy as jnp
from bokeh.palettes import plasma
from bokeh.models import HoverTool
from bokeh.models import BoxSelectTool
from bokeh.models import Circle
from bokeh.models import Range1d
from cml.plot import cml_figure, cml_figure_legend
from bokeh.models import ColumnDataSource

def scatter(
        x: jnp.ndarray,
        y: jnp.ndarray,
        title: str = r"Observations",
        toolbar_location:str = None,
        x_label:str = "x",
        y_label:str = "y",
        legend_labels="Observations"):
    # Create a figure in our style
    p = cml_figure(
        plot_width=600, 
        plot_height=450, 
        toolbar_location=toolbar_location, 
        title=title)
    # Display the circles
    cr = p.circle(
        np.array(x), 
        np.array(y), 
        size = 10,
        fill_color="midnightblue", 
        alpha = 0.25, 
        line_color = "white",
        hover_fill_color = "red", 
        hover_alpha=0.5,
        hover_line_color="white",
        legend_label=legend_labels)
    # Annotate labels
    p.xaxis.axis_label = x_label
    p.yaxis.axis_label = y_label
    # Put hover tool
    p.add_tools(HoverTool(tooltips=None, renderers=[cr], mode='vline'))
    # Put selection tool
    cr.selection_glyph = Circle(fill_color="red", fill_alpha=0.8, line_color="white")
    cr.nonselection_glyph = Circle(fill_color="midnightblue", fill_alpha=0.5, line_color="white")
    p.add_tools(BoxSelectTool(renderers=[cr], mode='append'))
    # Add the legend
    p = cml_figure_legend(p)
    return p

def scatter_classes(
        x: jnp.ndarray,
        y: jnp.ndarray,
        title: str = r"Observations",
        toolbar_location:str = None,
        x_label:str = "x",
        y_label:str = "y"):
    colors = ['midnightblue', 'firebrick', 'forestgreen', 'darkorchid', 'chocolate']
    symbols = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5']
    p = cml_figure(plot_width=700, plot_height=450, title=title)
    classes = [int(x) for x in np.array(y.astype(int))]
    source = ColumnDataSource(dict(
        x=np.array(x[:, 0]),
        y=np.array(x[:, 1]),
        color=[colors[c] for c in classes],
        label=[symbols[c] for c in classes]
    ))
    p.circle( x='x', y='y', color='color', legend_field='label', source=source, size=10, alpha=0.6, line_color="white")
    
    #cr.selection_glyph = Circle(fill_color="red", fill_alpha=0.8, line_color="white")
    #cr.nonselection_glyph = Circle(fill_color="midnightblue", fill_alpha=0.5, line_color="white")
    p.xaxis.axis_label = x_label
    p.yaxis.axis_label = y_label
    p.legend.border_line_width = 3
    p.legend.border_line_color = "grey"
    p.legend.border_line_alpha = 0.8
    p.legend.background_fill_color = "darkgrey"
    p.legend.background_fill_alpha = 0.2
    p.legend.label_text_color = "white"
    #p.add_tools(BoxSelectTool(renderers=[cr], mode='append'))
    p.add_layout(p.legend[0], 'right')
    return p

# Plot classification with evolving boundary
def scatter_boundary(
        x: jnp.ndarray,
        y: jnp.ndarray,
        weights: jnp.ndarray,
        bias: jnp.ndarray,
        n_iter: int = 50,
        title: str = r"Observations",
        toolbar_location:str = None,
        x_label:str = "x",
        y_label:str = "y"
        ):
    grad_color = plasma(n_iter)
    fig = scatter_classes(x, y, title, toolbar_location, x_label, y_label)
    x_min, x_max = np.min(x[:, 0]), np.max(x[:, 0])
    y_min, y_max = np.min(x[:, 1]), np.max(x[:, 1])
    x_np = np.array([x_min, x_max])
    fig.x_range=Range1d(x_min, x_max)
    fig.y_range=Range1d(y_min, y_max)
    for i in range(n_iter):
        fig.line(x_np, (-np.dot(weights[i, 0], x_np) - bias[i]) / weights[i, 1], line_dash='dashed', line_width=4, color=grad_color[i]);
    return fig