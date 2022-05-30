"""Iterative Gaussian Process Approximation Methods."""

from ._auto_precond_conjugate_gradient import AutoPreconditionedConjugateGradient
from ._cholesky import Cholesky
from ._conjugate_gradient import ConjugateGradient
from ._mixed_strategy import MixedStrategy
from ._projected_bayes_regressor import ProjectedBayesRegressor
from ._pseudo_input import PseudoInput

# Aliases for convenience
CG = ConjugateGradient
AutoPCG = AutoPreconditionedConjugateGradient
PBR = ProjectedBayesRegressor

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "AutoPreconditionedConjugateGradient",
    "Cholesky",
    "ConjugateGradient",
    "MixedStrategy",
    "ProjectedBayesRegressor",
    "PseudoInput",
]

# Set correct module paths. Corrects links and module paths in documentation.
AutoPreconditionedConjugateGradient.__module__ = "itergp.methods"
Cholesky.__module__ = "itergp.methods"
ConjugateGradient.__module__ = "itergp.methods"
MixedStrategy.__module__ = "itergp.methods"
ProjectedBayesRegressor.__module__ = "itergp.methods"
PseudoInput.__module__ = "itergp.methods"
