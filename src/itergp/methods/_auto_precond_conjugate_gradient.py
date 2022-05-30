"""Auto-preconditioned conjugate gradient method."""

from typing import Tuple

from probnum import linalg
from probnum.linalg.solvers import (
    ProbabilisticLinearSolver,
    information_ops,
    stopping_criteria,
)

from itergp.methods import belief_updates, policies


class AutoPreconditionedConjugateGradient(ProbabilisticLinearSolver):
    r"""Auto-Preconditioned Conjugate Gradient method.

    Linear solver which initially chooses actions from a ``precond_policy`` and then
    constructs a preconditioner from the collected information to take preconditioned
    gradient actions afterwards.

    Parameters
    ----------
    precond_iter
        Number of iterations to use for preconditioner construction. Equivalently, the
        rank of the low-rank component of the preconditioner (inverse).
    precond_policy
        Policy to use for the first ``precond_size`` iterations, which determines the
        constructed preconditioner. Defaults to a
        :class:`~itergp.methods.policies.UnitVectorPolicy`, which is equivalent to
        preconditioning with a partial Cholesky factorization.
    maxiter
        Maximum number of steps the solver should take. Defaults :math:`10n`,
        where :math:`n` is the size of the linear system.
    atol :
        Absolute tolerance.
    rtol :
        Relative tolerance.
    """

    def __init__(
        self,
        precond_iter: int,
        precond_policy: Tuple[
            linalg.solvers.policies.LinearSolverPolicy
        ] = policies.UnitVectorPolicy(),
        maxiter: int = None,
        atol: float = 1e-6,
        rtol: float = 1e-6,
    ):
        super().__init__(
            policy=policies.AutoPreconditionedGradientPolicy(
                precond_iter=precond_iter, precond_policy=precond_policy
            ),
            information_op=information_ops.ProjectedResidualInformationOp(),
            belief_update=belief_updates.ProjectedResidualBeliefUpdate(),
            stopping_criterion=stopping_criteria.MaxIterationsStoppingCriterion(
                maxiter=maxiter
            )
            | stopping_criteria.ResidualNormStoppingCriterion(atol=atol, rtol=rtol),
        )
