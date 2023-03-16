"""
 ~ CML // Creative Machine Learning ~
 panel.py : Simple helper functions for Panel styling
 
 For the interactive GUI aspects, we rely on the Panel library
 
 Author               :  Philippe Esling
                        <esling@ircam.fr>
"""

import param
import panel as pn

def initialize_panel():
    pn.extension('ace', css_files=['cml/cml.css'])