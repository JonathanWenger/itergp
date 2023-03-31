"""Policies of probabilistic linear solvers returning actions."""

from ._adverserial_policy import AdverserialPolicy
from ._auto_preconditioned_gradient import AutoPreconditionedGradientPolicy
from ._conjugate_gradient import ConjugateGradientPolicy
from ._eigenvector import EigenvectorPolicy
from ._gradient import GradientPolicy
from ._mixed import MixedPolicy
from ._pseudo_input import PseudoInputPolicy
from ._unit_vector import UnitVectorPolicy

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "AdverserialPolicy",
    "AutoPreconditionedGradientPolicy",
    "ConjugateGradientPolicy",
    "EigenvectorPolicy",
    "GradientPolicy",
    "MixedPolicy",
    "PseudoInputPolicy",
    "UnitVectorPolicy",
]

# Set correct module paths. Corrects links and module paths in documentation.
AdverserialPolicy.__module__ = "itergp.methods.policies"
AutoPreconditionedGradientPolicy.__module__ = "itergp.methods.policies"
EigenvectorPolicy.__module__ = "itergp.methods.policies"
ConjugateGradientPolicy.__module__ = "itergp.methods.policies"
GradientPolicy.__module__ = "itergp.methods.policies"
UnitVectorPolicy.__module__ = "itergp.methods.policies"
PseudoInputPolicy.__module__ = "itergp.methods.policies"
MixedPolicy.__module__ = "itergp.methods.policies"
