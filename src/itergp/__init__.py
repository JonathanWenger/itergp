"""IterGP: A Novel Iterative Framework for Gaussian Process Approximation."""

# isort: off
from . import linops
from . import kernels

# isort: on
from . import datasets, methods
from ._gaussian_process import ConditionalGaussianProcess, GaussianProcess
from ._version import version as __version__

# Public classes and functions. Order is reflected in documentation.
__all__ = ["GaussianProcess", "ConditionalGaussianProcess"]

# Set correct module paths. Corrects links and module paths in documentation.
GaussianProcess.__module__ = "itergp"
ConditionalGaussianProcess.__module__ = "itergp"
