"""Policy returning gradient / residual actions."""

from __future__ import annotations

from typing import Optional

from probnum import backend, linops
from probnum.backend.random import RNGState
from probnum.linalg.solvers import policies
from probnum.typing import LinearOperatorLike


class GradientPolicy(policies.LinearSolverPolicy):
    """Gradient / residual policy.

    Policy returning gradients / residuals :math:`b - Ax_i` as actions. If the inverse
    of a preconditioner is supplied, returns :math:`P^{-1}(b-Ax_i)` instead.

    Parameters
    ----------
    precond_inv
        Inverse :math:`P^{-1}` of the preconditioner.
    """

    def __init__(self, precond_inv: Optional[LinearOperatorLike] = None) -> None:
        if precond_inv is not None:
            self._precond_inv = linops.aslinop(precond_inv)
        else:
            self._precond_inv = precond_inv
        super().__init__()

    @property
    def precond_inv(self) -> Optional[linops.LinearOperator]:
        """Preconditioner inverse."""
        return self._precond_inv

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
        if self._precond_inv is None:
            return solver_state.residual

        return self._precond_inv @ solver_state.residual
