"""Covariance functions / Kernels."""
from __future__ import annotations

# Public classes and functions. Order is reflected in documentation.
__all__ = ["ExpQuad", "Matern", "ScaledKernel", "WhiteNoise", "SumKernel"]

from typing import Optional

from probnum import backend, linops
from probnum.backend.typing import ArrayLike
from probnum.randprocs.kernels import ExpQuad, Matern, WhiteNoise
from probnum.randprocs.kernels._arithmetic_fallbacks import ScaledKernel, SumKernel

try:
    from pykeops.numpy import LazyTensor, Pm, Vi, Vj
except ImportError as e:
    pass


from itergp import linops


def kernel_matrix_linop(
    self,
    x0: ArrayLike,
    x1: Optional[ArrayLike] = None,
) -> backend.Array:
    """A convenience function for computing a kernel matrix linear operator for two sets
    of inputs.

    Parameters
    ----------
    x0
        *shape=* ``(M,) +`` :attr:`input_shape` or :attr:`input_shape`
        -- Stack of inputs for the first argument of the :class:`Kernel`.
    x1
        *shape=* ``(N,) +`` :attr:`input_shape` or :attr:`input_shape`
        -- (Optional) stack of inputs for the second argument of the
        :class:`Kernel`. If ``x1`` is not specified, the function behaves as if
        ``x1 = x0`` (but it is implemented more efficiently).

    Raises
    ------
    ValueError
        If the shapes of the inputs don't match the specification.
    """

    x0 = backend.asarray(x0)
    x1 = x0 if x1 is None else backend.asarray(x1)

    # Shape checking
    errmsg = (
        "`{argname}` must have shape `({batch_dim},) + input_shape` or "
        f"`input_shape`, where `input_shape` is `{self.input_shape}`, but an array "
        "with shape `{shape}` was given."
    )

    if not 0 <= x0.ndim - self._input_ndim <= 1:
        raise ValueError(errmsg.format(argname="x0", batch_dim="M", shape=x0.shape))

    if not 0 <= x1.ndim - self._input_ndim <= 1:
        raise ValueError(errmsg.format(argname="x1", batch_dim="N", shape=x1.shape))

    return linops.KernelMatrix(kernel=self, x0=x0, x1=x1)


def _expquad_keops_lazy_tensor(
    self,
    x0: ArrayLike,
    x1: Optional[ArrayLike] = None,
) -> LazyTensor:
    return (
        -Pm(backend.asarray([0.5 / self.lengthscale**2])) * Vi(x0).sqdist(Vj(x1))
    ).exp()


def _matern_keops_lazy_tensor(
    self,
    x0: ArrayLike,
    x1: Optional[ArrayLike] = None,
) -> LazyTensor:
    squared_dists = Vi(x0).sqdist(Vj(x1))
    scaled_dists = backend.sqrt(2 * self.nu) / self.lengthscale * squared_dists.sqrt()

    if self.nu == 0.5:
        return (-scaled_dists).exp()
    if self.nu == 1.5:
        return (1.0 + scaled_dists) * (-scaled_dists).exp()
    if self.nu == 2.5:
        return (1.0 + scaled_dists + scaled_dists**2 / 3.0) * (-scaled_dists).exp()
    if self.nu == 3.5:
        return (
            1.0
            + (1.0 + (2.0 / 5.0 + scaled_dists / 15.0) * scaled_dists) * scaled_dists
        ) * (-scaled_dists).exp()
    else:
        raise NotImplementedError


def _scaled_keops_lazy_tensor(
    self,
    x0: ArrayLike,
    x1: Optional[ArrayLike] = None,
) -> LazyTensor:
    return self._scalar * self._kernel._keops_lazy_tensor(x0=x0, x1=x1)


def _sum_keops_lazy_tensor(
    self,
    x0: ArrayLike,
    x1: Optional[ArrayLike] = None,
) -> LazyTensor:
    return self._kernel._summands[0]._keops_lazy_tensor(
        x0=x0, x1=x1
    ) + self._kernel._summands[1]._keops_lazy_tensor(x0=x0, x1=x1)


Matern.linop = kernel_matrix_linop
ExpQuad.linop = kernel_matrix_linop
ScaledKernel.linop = kernel_matrix_linop
Matern._keops_lazy_tensor = _matern_keops_lazy_tensor
ExpQuad._keops_lazy_tensor = _expquad_keops_lazy_tensor
ScaledKernel._keops_lazy_tensor = _scaled_keops_lazy_tensor
SumKernel._keops_lazy_tensor = _sum_keops_lazy_tensor
