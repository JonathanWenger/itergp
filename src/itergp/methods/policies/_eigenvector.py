"""Policy returning eigenvectors."""

from __future__ import annotations

from typing import Optional

from probnum import backend, linops
from probnum.backend.random import RNGState
from probnum.linalg.solvers import policies


class EigenvectorPolicy(policies.LinearSolverPolicy):
    """Eigenvector policy.

    Policy returning eigenvectors of the system matrix.

    Parameters
    ----------
    descending
        Whether to return eigenvectors in descending or ascending order of eigenvalues.
    """

    def __init__(self, descending: bool = True) -> None:
        self._descending = descending

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
        if not "eigenvectors" in solver_state.cache.keys():
            if isinstance(solver_state.problem.A, linops.LinearOperator):
                A = solver_state.problem.A.todense()
            else:
                A = solver_state.problem.A
            eigvals, eigvectors = backend.linalg.eigh(A)
            idx = backend.argsort(eigvals, descending=self.descending)
            solver_state.cache["eigenvectors"] = eigvectors[..., idx]

        action = solver_state.cache["eigenvectors"][..., solver_state.step]

        return action

    @property
    def descending(self) -> bool:
        """Whether the returned eigenvectors are returned in descending or ascending
        order of the eigenvalues."""
        return self._descending
