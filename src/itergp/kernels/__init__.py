"""Covariance functions / Kernels."""
from __future__ import annotations

# Public classes and functions. Order is reflected in documentation.
__all__ = ["ExpQuad", "Matern", "ScaledKernel"]

from typing import Optional

from probnum import backend
from probnum.backend.typing import ArrayLike
from probnum.randprocs.kernels import ExpQuad, Matern
from probnum.randprocs.kernels._arithmetic_fallbacks import ScaledKernel
from pykeops.numpy import LazyTensor, Pm, Vi, Vj

from itergp import linops


def linop(
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
    sq_distances = Vi(x0).sqdist(Vj(x1))
    distances = sq_distances.sqrt()

    if self.nu == 0.5:
        return (-Pm(backend.asarray([1.0 / self.lengthscale])) * distances).exp()

    if self.nu == 1.5:
        scaled_distances = (
            Pm(backend.asarray([backend.sqrt(3) / self.lengthscale])) * distances
        )
        return (1.0 + scaled_distances) * (-scaled_distances).exp()
    else:
        raise NotImplementedError


def _scaled_keops_lazy_tensor(
    self,
    x0: ArrayLike,
    x1: Optional[ArrayLike] = None,
) -> LazyTensor:
    return self._scalar * self._kernel._keops_lazy_tensor(x0=x0, x1=x1)


Matern.linop = linop
ExpQuad.linop = linop
ScaledKernel.linop = linop
Matern._keops_lazy_tensor = _matern_keops_lazy_tensor
ExpQuad._keops_lazy_tensor = _expquad_keops_lazy_tensor
ScaledKernel._keops_lazy_tensor = _scaled_keops_lazy_tensor
