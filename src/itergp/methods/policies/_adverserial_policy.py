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
    base_policy
        Policy which generates :math:`\tilde{s}_i` which then are transformed to be
        orthogonal to :math:`y`.
    """

    def __init__(self, base_policy: policies.LinearSolverPolicy) -> None:
        self._base_policy = base_policy
        super().__init__()

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
            action = self._base_policy(solver_state=solver_state, rng=rng)

            # Enforce orthogonality to observations
            return action - y * (y.T @ action) / (y.T @ y)

        return y

    @property
    def base_policy(self) -> policies.LinearSolverPolicy:
        return self._base_policy
