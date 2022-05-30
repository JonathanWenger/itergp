"""Policy returning actions from multiple different policies."""

from __future__ import annotations

import bisect
from typing import Optional, Tuple

from probnum import backend
from probnum.backend.random import RNGState
from probnum.linalg.solvers import policies


class MixedPolicy(policies.LinearSolverPolicy):
    """Mixed policy.

    Policy which chooses actions based on a set of base policies.

    Parameters
    ----------
    base_policies
        Policies which make up the :class:`MixedPolicy`.
    iters
        Until which iteration (non-inclusive) to use the policy in the corresponding
        position in ``base_policies``. Assumed to be sorted in increasing order. If
        ``iters`` has one fewer entry than ``base_policies``, the last policy is used
        for all remaining iterations.
    """

    def __init__(
        self, base_policies: Tuple[policies.LinearSolverPolicy], iters: Tuple[int]
    ) -> None:
        self._base_policies = base_policies
        self._iters = iters
        super().__init__()

    def __call__(
        self,
        solver_state: "probnum.linalg.solvers.LinearSolverState",
        rng: Optional[RNGState] = None,
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
        policy_idx = bisect.bisect_right(self._iters, solver_state.step)
        return self._base_policies[policy_idx](solver_state=solver_state, rng=rng)
