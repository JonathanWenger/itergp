"""Linear operator representing a kernel matrix."""

from __future__ import annotations

from typing import Optional
import warnings

from probnum import backend, linops, randprocs

_USE_KEOPS = True
try:
    import pykeops
except ImportError:
    _USE_KEOPS = False
    warnings.warning("KeOps is not installed and currently unavailable for Windows. This may prevent scaling to large datasets.")

class KernelMatrix(linops.LinearOperator):
    r"""Kernel matrix.

    Linear operator defining (matrix-)multiplication with a kernel matrix
    :math:`K=k(X_0,X_1) \in \mathbb{R}^{n_0 \times n_1}`, constructed from two sets of
    inputs :math:`X_i \in \mathbb{R}^{n_i \times d}`.

    Parameters
    ----------
    kernel
        Kernel.
    x0
        Inputs for the first argument of ``kernel``.
    x1
        Inputs for the second argument of ``kernel``.
    size_keops
        At what size of the training data to use KeOps ``LazyTensor`` instead of full
        matrices in memory.
    """

    def __init__(
        self,
        kernel: randprocs.kernels.Kernel,
        x0: backend.Array,
        x1: Optional[backend.Array] = None,
        size_keops=1000,
    ):

        self._kernel = kernel

        self._x0 = x0 = backend.asarray(x0)
        self._x1 = x0 if x1 is None else backend.asarray(x1)
        self._use_keops = _USE_KEOPS and (x0.shape[0] >= size_keops)
            

        super().__init__(
            shape=(self._x0.shape[0], self._x1.shape[0]),
            dtype=backend.promote_types(self._x0.dtype, self._x1.dtype),
            matmul=self._matmul,
            rmatmul=self._rmatmul,
            todense=lambda: self._kernel.matrix(self._x0, self._x1),
            transpose=lambda: KernelMatrix(kernel=kernel, x0=x1, x1=x0),
        )

        # Matrix properties
        if x1 is None:
            self.is_symmetric = True
            self.is_positive_definite = True

    @property
    def kernel(self) -> randprocs.kernels.Kernel:
        """Covariance function of the kernel matrix."""
        return self._kernel

    @property
    def x0(self) -> backend.Array:
        """First input(s)."""
        return self._x0

    @property
    def x1(self) -> backend.Array:
        """Second input(s)."""
        return self._x1

    def _matmul(self, x: backend.Array):
        if self._use_keops:
            x0 = self._x0
            x1 = self._x1
            if self._kernel.input_ndim == 0:
                if self._x0.ndim == 1:
                    x0 = self._x0[:, None]
                if self._x1.ndim == 1:
                    x1 = self._x1[:, None]

            return (
                self._kernel._keops_lazy_tensor(x0, x1) @ x
            )  # pylint: disable=protected-access

        return self.todense() @ x

    def _rmatmul(self, x: backend.Array):
        if self._use_keops:
            return (self._matmul(x.T)).T
        return x @ self.todense()
