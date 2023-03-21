"""Test fixtures for linear operators."""
from typing import Optional

from probnum import backend, randprocs
from probnum.backend.typing import ShapeType

from itergp import kernels

import pytest
import pytest_cases


@pytest_cases.fixture(scope="package")
@pytest_cases.parametrize("shape", [(), (1,), (10,)], idgen="input_shape{shape}")
def input_shape(shape: ShapeType) -> ShapeType:
    """Input dimension of the random process."""
    return shape


@pytest_cases.fixture(scope="package")
@pytest_cases.parametrize("shape", [()], idgen="output_shape{shape}")
def output_shape(shape: ShapeType) -> ShapeType:
    """Output dimension of the random process."""
    return shape


# Kernels
@pytest.fixture(
    params=[
        pytest.param(kerndef, id=kerndef[0].__name__)
        for kerndef in [
            (kernels.ExpQuad, {"lengthscale": 1.5}),
            (kernels.Matern, {"lengthscale": 1.0, "nu": 0.5}),
            (kernels.Matern, {"lengthscale": 1.0, "nu": 1.5}),
            (kernels.Matern, {"lengthscale": 1.0, "nu": 2.5}),
            (kernels.Matern, {"lengthscale": 1.0, "nu": 3.5}),
        ]
    ],
)
def kernel(request, input_shape: ShapeType) -> randprocs.kernels.Kernel:
    """Kernel / covariance function."""
    return request.param[0](input_shape=input_shape, **request.param[1])


# Test data for "linops.KernelMatrix"
@pytest.fixture(
    params=[
        pytest.param(shape, id=f"x0{shape}")
        for shape in [
            (1,),
            (2,),
            (10,),
        ]
    ],
)
def x0_batch_shape(request) -> ShapeType:
    """Batch shape of the first argument of ``Kernel.matrix``."""
    return request.param


@pytest.fixture(
    params=[
        pytest.param(shape, id=f"x1{shape}")
        for shape in [
            None,
            (1,),
            (3,),
            (10,),
        ]
    ],
)
def x1_batch_shape(request) -> Optional[ShapeType]:
    """Batch shape of the second argument of ``Kernel.matrix`` or ``None`` if the second
    argument is ``None``."""
    return request.param


@pytest.fixture()
def x0(x0_batch_shape: ShapeType, input_shape: ShapeType) -> backend.Array:
    """Random data from a standard normal distribution."""
    rng_state = backend.random.rng_state(seed=42)
    return backend.random.standard_normal(
        rng_state=rng_state, shape=x0_batch_shape + input_shape
    )


@pytest.fixture()
def x1(
    x1_batch_shape: Optional[ShapeType],
    input_shape: ShapeType,
) -> Optional[backend.Array]:
    """Random data from a standard normal distribution."""
    if x1_batch_shape is None:
        return None

    rng_state = backend.random.rng_state(seed=42)
    return backend.random.standard_normal(
        rng_state=rng_state, shape=x1_batch_shape + input_shape
    )
