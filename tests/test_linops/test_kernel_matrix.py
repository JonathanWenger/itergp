"""Tests for the kernel matrix linear operator."""

from probnum import backend, compat

from itergp import linops

from pytest_cases import parametrize_with_cases


@parametrize_with_cases(
    "linop",
    cases=".cases.kernel_matrix_cases",
)
def test_matvec_equal_to_dense_matvec(linop: linops.KernelMatrix):
    x = backend.ones(shape=(linop.shape[1],))
    compat.testing.assert_allclose(linop.todense() @ x, linop @ x)
