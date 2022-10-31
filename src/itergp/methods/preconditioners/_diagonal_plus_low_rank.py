"""Diagonal-plus-low-rank preconditioner."""

from __future__ import annotations

from typing import Union

from probnum import backend, linops
from probnum.linalg import solvers
from probnum.typing import LinearOperatorLike

import itergp


class DiagonalPlusLowRank(linops.LinearOperator):
    r"""Diagonal-plus-low-rank preconditioner.

    Preconditioner consisting of a diagonal component
    :math:`\Lambda \in \mathbb{R}^{n \times n}` and a low-rank component of rank
    :math:`k`, given by :math:`P = \Lambda + U U^\top`, where
    :math:`U \in \mathbb{R}^{n \times k}`.

    Parameters
    ----------
    diagonal
        Diagonal component.
    low_rank_factor
        Factor :math:`U` of the low-rank component :math:`UU^\top`.
    """

    def __init__(
        self,
        diagonal: Union[backend.Scalar, backend.Array, linops.Scaling],
        low_rank_factor: LinearOperatorLike,
    ) -> None:

        # Components
        if not isinstance(diagonal, linops.Scaling):
            self._diagonal = linops.Scaling(
                factors=diagonal,
                shape=(low_rank_factor.shape[0], low_rank_factor.shape[0]),
            )
        else:
            self._diagonal = diagonal

        self._low_rank = itergp.linops.LowRankMatrix(U=low_rank_factor)

        sum_linop = self._diagonal + self._low_rank

        # Inverse via matrix inversion lemma
        small_matrix = linops.Identity(
            shape=(low_rank_factor.shape[1], low_rank_factor.shape[1]),
            dtype=sum_linop.dtype,
        ) + low_rank_factor.T @ (self.diag_comp.inv() @ low_rank_factor)
        small_matrix_cholfac = backend.linalg.cholesky(
            small_matrix.todense(), upper=False
        )

        def inv_matmul(X):
            diag_inv_X = self.diag_comp.inv() @ X
            return self.diag_comp.inv() @ X - self.diag_comp.inv() @ (
                low_rank_factor
                @ backend.linalg.solve_cholesky(
                    small_matrix_cholfac,
                    low_rank_factor.T @ diag_inv_X,
                    lower=True,
                )
            )

        diag_plus_low_rank_inverse = linops.LinearOperator(
            shape=self._diagonal.shape, dtype=self._diagonal.dtype, matmul=inv_matmul
        )

        super().__init__(
            shape=self._diagonal.shape,
            dtype=sum_linop.dtype,
            matmul=sum_linop.__matmul__,
            rmatmul=sum_linop.__rmatmul__,
            todense=sum_linop.todense,
            transpose=lambda: self,
            inverse=lambda: diag_plus_low_rank_inverse,
            rank=lambda: self._diagonal.shape[0]
            if self._diagonal.is_positive_definite
            else None,
            trace=sum_linop.trace,
        )

        # Matrix properties
        self.is_symmetric = True
        if self._diagonal.is_positive_definite:
            self.is_positive_definite = True

    @property
    def diag_comp(self) -> linops.Scaling:
        """Diagonal component."""
        return self._diagonal

    @property
    def low_rank_comp(self) -> itergp.linops.LowRankMatrix:
        """Low-rank component."""
        return self._low_rank

    @classmethod
    def from_kernel_matrix_linear_solve(
        cls,
        solver_state: solvers.LinearSolverState,
    ):
        kernel_linop = solver_state.problem.A._summands[0]
        noise_linop = solver_state.problem.A._summands[1]

        diagonal = backend.diagonal(noise_linop.todense())
        low_rank_factor = kernel_linop @ solver_state.belief.Ainv._summands[1].U

        return cls(diagonal=diagonal, low_rank_factor=low_rank_factor)
