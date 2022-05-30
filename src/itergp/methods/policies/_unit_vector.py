"""Policy returning unit vectors."""

from __future__ import annotations

from typing import Optional

from probnum import backend
from probnum.backend.random import RNGState
from probnum.linalg.solvers import policies


class UnitVectorPolicy(policies.LinearSolverPolicy):
    """Standard unit vector policy.

    Policy returning standard unit vectors according to a given ordering.

    Parameters
    ----------
    ordering
        Ordering strategy of the rows (and columns) of the system matrix.
    """

    def __init__(self, ordering: str = "lexicographic") -> None:
        self._ordering = ordering

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
        if not "ordering" in solver_state.cache.keys():
            if self.ordering == "lexicographic":
                solver_state.cache["ordering"] = backend.arange(
                    0, solver_state.problem.A.shape[0] + 1
                )
            else:
                # TODO: support other orderings
                raise NotImplementedError

        action = backend.zeros((solver_state.problem.A.shape[0],))
        action[solver_state.cache["ordering"][solver_state.step]] = 1.0

        return action

    @property
    def ordering(self) -> str:
        """Ordering strategy defining in which order to select datapoints."""
        return self._ordering
