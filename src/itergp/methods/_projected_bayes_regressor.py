"""Projected Bayes regressor."""

from probnum.linalg.solvers import (
    ProbabilisticLinearSolver,
    information_ops,
    stopping_criteria,
)

import itergp.methods
from itergp.methods import belief_updates, policies


class ProjectedBayesRegressor(ProbabilisticLinearSolver):
    """Projected Bayes Regressor.

    Parameters
    ----------
    descending
        Whether to pick eigenvectors in descending order of eigenvalues.
    maxiter
        Maximum number of steps the solver should take.
    atol :
        Absolute tolerance.
    rtol :
        Relative tolerance.
    """

    def __init__(
        self,
        descending: bool = True,
        maxiter: int = None,
        atol: float = 1e-6,
        rtol: float = 1e-6,
    ):
        super().__init__(
            policy=policies.EigenvectorPolicy(descending=descending),
            information_op=information_ops.ProjectedResidualInformationOp(),
            belief_update=belief_updates.ProjectedResidualBeliefUpdate(),
            stopping_criterion=itergp.methods.stopping_criteria.MaxIterationsStoppingCriterion(
                maxiter=maxiter, problem_size_factor=1
            )
            | stopping_criteria.ResidualNormStoppingCriterion(atol=atol, rtol=rtol),
        )
