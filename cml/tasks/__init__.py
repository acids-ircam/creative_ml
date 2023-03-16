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

from cml.tasks.task import (
    MLTask
)
from cml.tasks.regression import (
    RegressionTask,
    RegressionPolynomial,
    RegressionPolynomialSolver,
    RegressionSwissRoll
)
from cml.tasks.classification import (
    ClassificationTask,
    ClassificationLinear,
    ClassificationXOR,
    ClassificationTwoCircles,
    ClassificationTwoMoons,
    ClassificationBlobs
)
from cml.tasks.density import (
    DensityGaussian,
    DensityGaussian2D
)

__all__ = [
    "MLTask",
    "RegressionTask",
    "RegressionPolynomial",
    "RegressionPolynomialSolver",
    "RegressionSwissRoll",
    "ClassificationTask",
    "ClassificationLinear",
    "ClassificationXOR",
    "ClassificationTwoCircles",
    "ClassificationTwoMoons",
    "ClassificationBlobs",
    "DensityGaussian",
    "DensityGaussian2D"
]
