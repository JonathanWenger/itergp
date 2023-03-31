"""Policy returning unit vectors."""

from __future__ import annotations

from typing import Optional

from probnum import backend
from probnum.backend.random import RNGState
from probnum.linalg.solvers import policies


class AdverserialPolicy(policies.LinearSolverPolicy):
    r"""Adverserial policy.

    Policy returning actions which are orthogonal to the data causing no update to the
    posterior mean except for the last iteration. The actions are given by
    :math:`s_i = (I - y(y^\top y)^{-1}y^\top)\tilde{s}_i`, where
    :math:`\tilde{s}_i` are arbitrary linearly independent vectors.

    Parameters
    ----------

    """

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
        n = solver_state.problem.A.shape[0]
        y = solver_state.problem.b

        if solver_state.step < n - 1:
            # Arbitrary linearly independent action sequence
            action = backend.zeros((n,))
            action[solver_state.step] = 1.0

            # Enforce orthogonality to observations
            return action - y * (y.T @ action) / (y.T @ y)

        return y
