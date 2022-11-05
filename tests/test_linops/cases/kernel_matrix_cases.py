"""Cases defining kernel matrix linear operators."""
from probnum import backend, randprocs

from itergp import linops

from pytest_cases import case


@case(tags=["symmetric", "positive_definite"])
def case_kernel_matrix_linop(
    kernel: randprocs.kernels.Kernel, x0: backend.Array, x1: backend.Array
) -> linops.KernelMatrix:
    return linops.KernelMatrix(kernel=kernel, x0=x0, x1=x1)


@case(tags=["symmetric", "positive_definite"])
def case_kernel_matrix_linop_keops(
    kernel: randprocs.kernels.Kernel, x0: backend.Array, x1: backend.Array
) -> linops.KernelMatrix:
    return linops.KernelMatrix(kernel=kernel, x0=x0, x1=x1, size_keops=0)
