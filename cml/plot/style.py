"""
 ~ CML // Creative Machine Learning ~
 plot/style.py : Simple helper functions for plot styling
 
 Here we define simple handlers for data plotting in notebooks.
 We rely both on the Bokeh, Plotly and Matplotlib backends
 
 Author               :  Philippe Esling
                        <esling@ircam.fr>
"""

from bokeh.plotting import figure
import matplotlib as mpl
import matplotlib.pyplot as plt

def cml_figure(
        plot_width: int = 600, 
        plot_height: int = 450, 
        **kwargs) -> figure:
    """
    

    Parameters
    ----------
    plot_width : TYPE, optional
        DESCRIPTION. The default is 600.
    plot_height : TYPE, optional
        DESCRIPTION. The default is 450.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    p : TYPE
        DESCRIPTION.

    """
    # Create a Bokeh figure
    p = figure(plot_width=plot_width, plot_height=plot_height, **kwargs)
    # Background properties
    p.background_fill_color = '#0f0f0f'
    p.background_fill_alpha = 0.9
    # Border properties
    p.border_fill_color = '#111111'
    p.border_fill_alpha = 0.8
    # Outline properties
    p.outline_line_color = '#E0E0E0'
    p.outline_line_alpha = 0.25
    p.outline_line_width = 3
    # Grid properties
    p.grid.grid_line_color = '#E0E0E0'
    p.grid.grid_line_alpha = 0.15
    # Axis (ticks) properties
    p.axis.major_tick_line_alpha = 0.4
    p.axis.major_tick_line_color = '#E0E0E0'
    p.axis.minor_tick_line_alpha = 0.4
    p.axis.minor_tick_line_color = '#E0E0E0'
    # Axis (lines) properties
    p.axis.axis_line_alpha = 0.4
    p.axis.axis_line_color = '#E0E0E0'
    p.axis.major_label_text_color = '#E0E0E0'
    p.axis.major_label_text_font = 'Josefin Sans'
    p.axis.major_label_text_font_size = '1.15em'
    # Axis (labels) properties
    p.axis.axis_label_standoff = 10
    p.axis.axis_label_text_color = '#FFFFFF'
    p.axis.axis_label_text_font = 'Josefin Sans'
    p.axis.axis_label_text_font_size = '1.9em'
    p.axis.axis_label_text_font_style = 'bold'
    # Title properties
    p.title.text_color = '#E0E0E0'
    p.title.text_font = 'Josefin Sans'
    p.title.text_font_size = '2.1em'
    p.title.text_font_style = "bold"
    return p

def cml_figure_matplotlib(
        plot_width: int = 600, 
        plot_height: int = 450, 
        **kwargs) -> figure:
    """
    

    Parameters
    ----------
    plot_width : TYPE, optional
        DESCRIPTION. The default is 600.
    plot_height : TYPE, optional
        DESCRIPTION. The default is 450.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    p : TYPE
        DESCRIPTION.

    """
    plt.style.use('dark_background')
    mpl.rcParams.update({'font.size': 18, 'lines.linewidth': 3, 'lines.markersize': 15})
    # avoid type 3 (i.e. bitmap) fonts in figures
    mpl.rcParams['ps.useafm'] = True
    mpl.rcParams['pdf.use14corefonts'] = True
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['font.sans-serif'] = 'Helvetica'
    #mpl.rcParams['text.hinting'] = False
    # Set colors cycle
    colors = mpl.cycler('color', ['#3388BB', '#EE6666', '#9988DD', '#EECC55', '#88BB44', '#FFBBBB'])
    #plt.rc('figure', facecolor='#00000000', edgecolor='black')
    #plt.rc('axes', facecolor='#FFFFFF88', edgecolor='white', axisbelow=True, grid=True, prop_cycle=colors)
    plt.rc('legend', facecolor='#666666EE', edgecolor='white', fontsize=16)
    plt.rc('grid', color='white', linestyle='solid')
    plt.rc('text', color='white')
    plt.rc('xtick', direction='out', color='white')
    plt.rc('ytick', direction='out', color='white')
    plt.rc('patch', edgecolor='#E6E6E6')
    return plt.figure(figsize=(plot_height, plot_width))

def cml_figure_legend(p: figure) -> figure:
    p.legend.spacing = 8
    p.legend.glyph_width = 20    
    p.legend.label_standoff = 8    
    p.legend.label_text_color = '#E0E0E0'    
    p.legend.label_text_font = 'Josefin Sans'
    p.legend.label_text_font_size = '1.15em'
    p.legend.border_line_alpha = 0.25
    p.legend.background_fill_alpha = 0.25
    p.legend.background_fill_color = '#505050'
    return p
    
def cml_figure_colorbar(p: figure) -> figure:
    p.colorbar.title_text_color = '#E0E0E0'
    p.colorbar.title_text_font = 'Helvetica'
    p.colorbar.title_text_font_size = '1.025em'
    p.colorbar.title_text_font_style = 'normal'
    p.colorbar.major_label_text_color = '#E0E0E0'
    p.colorbar.major_label_text_font = 'Helvetica'
    p.colorbar.major_label_text_font_size = '1.025em'
    p.colorbar.background_fill_color = '#15191C'
    p.colorbar.major_tick_line_alpha = 0
    p.colorbar.bar_line_alpha = 0
    return p

def cml_figure_axis(p: figure) -> figure:
    p.linearaxis.major_tick_line_alpha = 0
    p.linearaxis.major_tick_line_color = '#E0E0E0'
    p.linearaxis.minor_tick_line_alpha = 0
    p.linearaxis.minor_tick_line_color = '#E0E0E0'
    p.linearaxis.axis_line_alpha = 0
    p.linearaxis.axis_line_color = '#E0E0E0'
    p.linearaxis.major_label_text_color = '#E0E0E0'
    p.linearaxis.major_label_text_font = 'Helvetica'
    p.linearaxis.major_label_text_font_size = '1.025em'
    p.linearaxis.axis_label_standoff = 10
    p.linearaxis.axis_label_text_color = '#E0E0E0'
    p.linearaxis.axis_label_text_font = 'Helvetica'
    p.linearaxis.axis_label_text_font_size = '1.25em'
    p.linearaxis.axis_label_text_font_style = 'normal'
    return p