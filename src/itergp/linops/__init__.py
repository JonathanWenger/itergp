"""Finite-dimensional linear operators """

from ._kernel_matrix import KernelMatrix
from ._low_rank import LowRankMatrix

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "KernelMatrix",
    "LowRankMatrix",
]
# Set correct module paths. Corrects links and module paths in documentation.
KernelMatrix.__module__ = "itergp.linops"
LowRankMatrix.__module__ = "itergp.linops"
