"""Preconditioners for linear systems.

Preconditioning accelerates and stabilizes the solution of linear systems.
A preconditioner :math:`P` is an approximation of a matrix :math:`A` such that the 
condition number of the matrix :math:`P^{-1}A` is lower than that of the original 
matrix.
"""

from ._diagonal_plus_low_rank import DiagonalPlusLowRank

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "DiagonalPlusLowRank",
]

# Set correct module paths. Corrects links and module paths in documentation.
DiagonalPlusLowRank.__module__ = "itergp.methods.preconditioners"
