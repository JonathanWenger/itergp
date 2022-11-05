"""Cases defining kernel matrix linear operators."""
from probnum import backend, randprocs

from itergp import linops


@case(tags=["symmetric", "positive_definite"])
def case_kernel_matrix_linop(
    kernel: randprocs.kernels.Kernel, x0: backend.Array, x1: backend.Array
) -> linops.KernelMatrix:
    return linops.KernelMatrix()
