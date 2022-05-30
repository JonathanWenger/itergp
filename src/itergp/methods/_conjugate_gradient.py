"""Conjugate Gradient Method."""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import probnum
from probnum import backend, linops
from probnum.linalg.solvers import (
    ProbabilisticLinearSolver,
    beliefs,
    information_ops,
    stopping_criteria,
)
from probnum.typing import LinearOperatorLike

from itergp.methods import belief_updates, policies


class ConjugateGradient(ProbabilisticLinearSolver):
    r"""Conjugate Gradient method.

    Parameters
    ----------
    precond_inv
        Preconditioner inverse.
    maxiter
        Maximum number of steps the solver should take. Defaults :math:`10n`,
        where :math:`n` is the size of the linear system.
    atol :
        Absolute tolerance.
    rtol :
        Relative tolerance.
    reorthogonalization_fn_residual
        Reorthogonalization function, which takes a vector, an orthogonal basis and
        optionally an inner product and returns a reorthogonalized vector. If not `None`
        the residuals are reorthogonalized before the action is computed.
    """

    def __init__(
        self,
        precond_inv: LinearOperatorLike = None,
        maxiter: int = None,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        reorthogonalization_fn_residual: Optional[
            Callable[
                [backend.Array, Iterable[backend.Array], linops.LinearOperator],
                backend.Array,
            ]
        ] = backend.linalg.gram_schmidt_double,
    ):
        # super().__init__(
        #     policy=policies.GradientPolicy(precond_inv=precond_inv),
        #     information_op=information_ops.ProjectedResidualInformationOp(),
        #     belief_update=belief_updates.ProjectedResidualBeliefUpdate(),
        #     stopping_criterion=stopping_criteria.MaxIterationsStoppingCriterion(
        #         maxiter=maxiter
        #     )
        #     | stopping_criteria.ResidualNormStoppingCriterion(atol=atol, rtol=rtol),
        # )
        super().__init__(
            policy=policies.ConjugateGradientPolicy(
                precond_inv=precond_inv,
                reorthogonalization_fn_residual=reorthogonalization_fn_residual,
            ),
            information_op=information_ops.ProjectedResidualInformationOp(),
            belief_update=ConjugateGradient.BeliefUpdate(),
            stopping_criterion=stopping_criteria.MaxIterationsStoppingCriterion(
                maxiter=maxiter
            )
            | stopping_criteria.ResidualNormStoppingCriterion(atol=atol, rtol=rtol),
        )

    class BeliefUpdate(belief_updates.ProjectedResidualBeliefUpdate):
        """Belief update of the conjugate gradient method."""

        def __call__(
            self, solver_state: "probnum.linalg.solvers.LinearSolverState"
        ) -> beliefs.LinearSystemBelief:

            # Search direction
            A_action = solver_state.problem.A @ solver_state.action
            search_dir = solver_state.action

            # Normalization constant
            gram = search_dir @ A_action
            gram_pinv = 1.0 / gram if gram > 0.0 else 0.0

            return self.updated_linsys_belief(
                search_dir=search_dir,
                A_search_dir=A_action,
                gram_pinv=gram_pinv,
                solver_state=solver_state,
            )
