"""
 ~ CML // Creative Machine Learning ~
 data/__init__.py : Initializing data generating functions
 
 In order to understand the fundamentals of ML, we will work on several toy
 datasets for different tasks that we are adressing
 - Regression
 
 Author               :  Philippe Esling
                        <esling@ircam.fr>
"""

from cml.data.audio import (
    AudioSupervisedDataset,
    import_dataset,
)
from cml.data.regression import (
    polynomial,
    linear,
    swiss_roll
)
from cml.data.classification import (
    linear_separation,
    xor_separation,
    two_circles,
    two_moons,
    make_blobs
)
from cml.data.density import (
    gaussian,
    bivariate,
    ring,
    wave,
    wave_twist,
    wave_split,
    circle,
    grid
)

__all__ = [
    "AudioSupervisedDataset",
    "import_dataset"
    "polynomial",
    "linear",
    "swiss_roll",
    "linear_separation",
    "xor_separation",
    "two_circles",
    "two_moons",
    "make_blobs",
    "gaussian",
    "bivariate",
    "ring",
    "wave",
    "wave_twist",
    "wave_split",
    "circle",
    "grid"
]
