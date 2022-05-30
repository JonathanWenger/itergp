"""Low-rank linear operator."""
from __future__ import annotations

from typing import Optional

from probnum import backend, linops


class LowRankMatrix(linops.LinearOperator):
    r"""Low-rank matrix.

    Linear operator given by a low rank matrix of the form

    .. math ::
            L = U V^\top

    where :math:`U, V \in \mathbb{R}^{n \times m}` are tall matrices
    with :math:`m < n`.

    Parameters
    ----------
    U
        First factor :math:`U \in \mathbb{R}^{n \times m}`.
    V
        Second factor :math:`V \in \mathbb{R}^{n \times m}`.
    """

    def __init__(
        self,
        U: backend.Array,
        V: Optional[backend.Array] = None,
    ):
        if U.ndim == 1:
            self._U = U.reshape(-1, 1)
        self._U = U

        if V is None:
            self._V = self._U
            self._identical_factors = True
        else:
            if V.ndim == 1:
                self._V = V.reshape(-1, 1)
            self._V = V

        self._linop = linops.aslinop(self._U) @ linops.aslinop(self._V.T)

        super().__init__(
            shape=self._linop.shape,
            dtype=self._linop.dtype,
            matmul=self._linop.__matmul__,
            rmatmul=self._linop.__rmatmul__,
            apply=self._linop.__call__,
            todense=self._linop.todense,
            transpose=lambda: self if V is None else lambda self: self._V @ self._U.T,
        )

        # Matrix properties
        if V is None:
            self.is_symmetric = True
            self.is_positive_definite = True

    @property
    def identical_factors(self) -> bool:
        return self._identical_factors

    @property
    def U(self) -> linops.LinearOperator:
        """First factor."""
        return self._U

    @property
    def V(self) -> linops.LinearOperator:
        """Second factor."""
        return self._V

    # TODO: Use this property to reduce a sum of low-rank linear operators with identical factors: UU' + uu' = [U u][U u]'
