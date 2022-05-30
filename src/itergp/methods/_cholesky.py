"""Cholesky Decomposition"""

import probnum
from probnum.linalg.solvers import ProbabilisticLinearSolver, information_ops

from itergp.methods import belief_updates, policies, stopping_criteria


class Cholesky(ProbabilisticLinearSolver):
    r"""Cholesky decomposition.

    Parameters
    ----------
    atol
        Absolute tolerance.
    rtol
        Relative tolerance.
    maxrank
        Maximum rank of the factorization.
    """

    def __init__(
        self,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        maxrank: int = None,
    ):
        super().__init__(
            policy=policies.UnitVectorPolicy(),
            information_op=information_ops.ProjectedResidualInformationOp(),
            belief_update=belief_updates.ProjectedResidualBeliefUpdate(),
            stopping_criterion=stopping_criteria.MaxIterationsStoppingCriterion(
                maxiter=maxrank, problem_size_factor=1
            )
            | probnum.linalg.solvers.stopping_criteria.ResidualNormStoppingCriterion(
                atol=atol, rtol=rtol
            ),
        )
