"""Policy returning kernel functions centered at inducing points."""

from __future__ import annotations

from typing import Optional

from probnum import backend
from probnum.backend.random import RNGState
from probnum.linalg.solvers import policies


class PseudoInputPolicy(policies.LinearSolverPolicy):
    """Pseudo-input policy.

    Parameters
    ----------
    pseudo_inputs
        Pseudo inputs, also known as inducing points.
    """

    def __init__(self, pseudo_inputs: backend.Array) -> None:
        self._pseudo_inputs = pseudo_inputs
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
        if not "kernel_linop" in solver_state.cache.keys():
            solver_state.cache["kernel_linop"] = solver_state.problem.A._summands[
                0
            ]  # TODO: make this more robust with a KernelLinearSystem(Belief)

        kernel = solver_state.cache["kernel_linop"].kernel
        X = solver_state.cache["kernel_linop"].x0
        action = kernel(X[:, None], self.pseudo_inputs[solver_state.step, ...])[:, 0]

        return action

    @property
    def pseudo_inputs(self) -> backend.Array:
        """Pseudo-inputs defining virtual observation locations."""
        return self._pseudo_inputs
