"""Small numerical compatibility entries shared by scientific services.

SciPy's trapezoid is available in the retained SciPy 1.7 environment and in
modern SciPy. It preserves the composite trapezoidal rule formerly requested
through NumPy's removed ``trapz`` name; no interpolation or resampling is added.
"""
from scipy.integrate import trapezoid

__all__ = ('trapezoid',)
