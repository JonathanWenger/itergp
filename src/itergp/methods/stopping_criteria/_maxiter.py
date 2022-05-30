"""Stopping criterion based on a maximum number of iterations."""
from __future__ import annotations

from typing import Optional

from probnum.linalg.solvers.stopping_criteria import LinearSolverStoppingCriterion


class MaxIterationsStoppingCriterion(LinearSolverStoppingCriterion):
    r"""Stop after a maximum number of iterations.

    Stop when the solver has taken a maximum number of steps. The maximum number of iterations
    can either be specified in absolute terms or relative to the problem size.

    Parameters
    ----------
    maxiter
        Maximum number of steps the solver should take.
    problem_size_factor
        If ``maxiter`` is None, stops after ``problem_size_factor * problem_size`` steps, where
        ``problem_size`` is the dimension of the solution to the linear system. Will be ignored
        otherwise.
    """

    def __init__(self, maxiter: Optional[int] = None, problem_size_factor: int = 10):
        self._maxiter = maxiter
        self._problem_size_factor = problem_size_factor

    def __call__(
        self, solver_state: "probnum.linalg.solvers.LinearSolverState"
    ) -> bool:
        """Check whether the maximum number of iterations has been reached.

        Parameters
        ----------
        solver_state
            Current state of the linear solver.
        """
        if self.maxiter is None:
            return (
                solver_state.step
                >= solver_state.problem.A.shape[0] * self.problem_size_factor
            )

        return solver_state.step >= self.maxiter

    @property
    def maxiter(self) -> int:
        """Maximum number of steps."""
        return self._maxiter

    @property
    def problem_size_factor(self) -> int:
        """Problem size factor.

        If ``maxiter`` is None, stops after ``problem_size_factor * problem_size`` steps, where
        ``problem_size`` is the dimension of the solution to the linear system. Will be ignored
        otherwise.
        """
        return self._problem_size_factor
