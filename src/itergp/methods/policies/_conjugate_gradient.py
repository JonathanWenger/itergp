"""Policy returning :math:`A`-conjugate actions."""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Tuple

import numpy as np
from probnum import backend, linops
from probnum.linalg.solvers import policies
from probnum.typing import LinearOperatorLike


class ConjugateGradientPolicy(policies.LinearSolverPolicy):
    r"""Policy returning :math:`A`-conjugate actions.

    Selects the negative gradient / residual as an initial action
    :math:`s_0 = b - A x_0` and then successively generates :math:`A`-conjugate actions,
    i.e. the actions satisfy :math:`s_i^\top A s_j = 0` iff :math:`i \neq j`. If a
    preconditioner inverse :math:`P^{-1}` is supplied, the actions are orthogonal with
    respect to the :math:`P^{-\frac{1}{2}}AP^{-\frac{\top}{2}}` inner product.

    Parameters
    ----------
    precond_inv
        Preconditioner inverse.
    reorthogonalization_fn_residual
        Reorthogonalization function, which takes a vector, an orthogonal basis and
        optionally an inner product and returns a reorthogonalized vector. If not `None`
        the residuals are reorthogonalized before the action is computed.
    """

    def __init__(
        self,
        precond_inv: Optional[LinearOperatorLike] = None,
        reorthogonalization_fn_residual: Optional[
            Callable[
                [backend.Array, Iterable[backend.Array], linops.LinearOperator],
                backend.Array,
            ]
        ] = None,
    ) -> None:
        if precond_inv is not None:
            self._precond_inv = linops.aslinop(precond_inv)
        else:
            self._precond_inv = None
        self._reorthogonalization_fn_residual = reorthogonalization_fn_residual

    def __call__(
        self,
        solver_state: "probnum.linalg.solvers.LinearSolverState",
        rng: Optional[np.random.Generator] = None,
    ) -> backend.Array:

        precond_inv = (
            self._precond_inv
            if self._precond_inv is not None
            else linops.Identity(shape=solver_state.problem.A.shape)
        )

        residual = solver_state.residual

        if solver_state.step == 0:
            if self._reorthogonalization_fn_residual is not None:
                solver_state.cache["reorthogonalized_residuals"].append(
                    solver_state.residual
                )

            return precond_inv @ residual
        else:
            # Reorthogonalization of the residual
            if self._reorthogonalization_fn_residual is not None:
                residual, prev_residual = self._reorthogonalized_residual(
                    solver_state=solver_state
                )
            else:
                prev_residual = solver_state.residuals[solver_state.step - 1]

            # Conjugacy correction (in exact arithmetic)
            beta = (
                residual.T
                @ precond_inv
                @ residual
                / (prev_residual.T @ precond_inv @ prev_residual)
            )
            return (
                precond_inv @ residual
                + beta * solver_state.actions[solver_state.step - 1]
            )

    def _reorthogonalized_residual(
        self,
        solver_state: "probnum.linalg.solvers.LinearSolverState",
    ) -> Tuple[backend.Array, backend.Array]:
        """Compute the reorthogonalized residual and its predecessor."""
        residual = self._reorthogonalization_fn_residual(
            v=solver_state.residual,
            orthogonal_basis=np.asarray(
                solver_state.cache["reorthogonalized_residuals"]
            ),
            inner_product=self._precond_inv,
        )
        solver_state.cache["reorthogonalized_residuals"].append(residual)
        prev_residual = solver_state.cache["reorthogonalized_residuals"][
            solver_state.step - 1
        ]
        return residual, prev_residual
