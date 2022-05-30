"""Stopping criteria for probabilistic linear solvers."""

from ._maxiter import MaxIterationsStoppingCriterion

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "MaxIterationsStoppingCriterion",
]

# Set correct module paths. Corrects links and module paths in documentation.
MaxIterationsStoppingCriterion.__module__ = "itergp.methods.stopping_criteria"
