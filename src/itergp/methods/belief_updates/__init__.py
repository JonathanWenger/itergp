"""Belief updates."""

from .projected_residual import ProjectedResidualBeliefUpdate

# Public classes and functions. Order is reflected in documentation.
__all__ = ["ProjectedResidualBeliefUpdate"]

# Set correct module paths. Corrects links and module paths in documentation.
ProjectedResidualBeliefUpdate.__module__ = "itergp.methods.belief_updates"
