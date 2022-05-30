"""Policy returning preconditioned gradients as actions from a self-constructed 
preconditioner."""

from __future__ import annotations

from typing import Optional, Tuple

from probnum import backend, linalg
from probnum.backend.random import RNGState

from itergp.methods import preconditioners

from ._gradient import GradientPolicy
from ._unit_vector import UnitVectorPolicy


class AutoPreconditionedGradientPolicy:
    """Policy which first constructs a preconditioner on the fly and then returns
    preconditioned gradient actions.

    This policy constructs a diagonal-plus-low-rank preconditioner by first using a
    ``precond_policy`` for ``precond_size`` steps. Afterwards the policy returns
    preconditioned gradients as actions using the previously constructed preconditioner.

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
    """

    def __init__(
        self,
        precond_iter: int,
        precond_policy: Tuple[
            linalg.solvers.policies.LinearSolverPolicy
        ] = UnitVectorPolicy(),
    ) -> None:

        self._precond_policy = precond_policy
        self._gradient_policy = None
        self._precond_iter = precond_iter

    def __call__(
        self,
        solver_state: "probnum.linalg.solvers.LinearSolverState",
        rng: Optional[RNGState],
    ) -> backend.Array:
        """Return an action for a given solver state.

        Parameters
        ----------
        solver_state
            Current state of the linear solver.
        rng
            Random number generator.

        Returns
        -------
        action
            Next action to take.
        """
        if solver_state.step < self._precond_iter:
            return self._precond_policy(solver_state=solver_state, rng=rng)
        elif solver_state.step == self._precond_iter:
            if self._precond_iter == 0:
                precond_inv = None
            else:
                precond = (
                    preconditioners.DiagonalPlusLowRank.from_kernel_matrix_linear_solve(
                        solver_state=solver_state
                    )
                )
                precond_inv = precond.inv()

            self._gradient_policy = GradientPolicy(precond_inv=precond_inv)

        return self._gradient_policy(solver_state=solver_state, rng=rng)
