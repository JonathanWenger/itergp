"""Synthetically generated dataset."""
from __future__ import annotations

from typing import Callable, Tuple

from probnum import backend
from probnum.backend.typing import ShapeLike, ShapeType

from ._dataset import Dataset


class SyntheticDataset(Dataset):
    """Synthetically generated dataset.

    Generate a dataset from random input data, a ground truth function and additive
    noise.

    Parameters
    ----------
    rng_state
        State of the random number generator.
    size
        Size of the dataset. Tuple of training and test dataset size.
    input_shape
        Shape of the inputs.
    output_shape
        Shape of the outputs.
    fun
        Ground truth function generating the data (without noise).
    noise_std
        Standard deviation of the Gaussian noise added to the value of :attr:`true_fn`.
    range_X
        Maximum and minimum value for the generated inputs.

    Examples
    --------
    >>> from probnum import backend
    >>> from itergp import datasets
    ...
    >>> rng_state = backend.random.rng_state(42)
    >>> data = datasets.SyntheticDataset(rng_state=rng_state, size=(100, 10), input_shape=2)
    >>> data.train.X.shape
    (100, 2)
    """

    def __init__(
        self,
        rng_state: backend.random.RNGState,
        size: Tuple[int, int],
        input_shape: ShapeLike = (),
        output_shape: ShapeLike = (),
        fun: Callable[[backend.Array], backend.Array] = lambda X: backend.sin(
            backend.pi * X
        )
        if X.ndim == 1
        else backend.sin(backend.pi * X @ backend.ones((X.shape[1:]))),
        noise_var: float = 0.1,
        range_X: Tuple[float, float] = [-1.0, 1.0],
    ):
        self._fun = fun
        input_shape = backend.asshape(input_shape)
        output_shape = backend.asshape(output_shape)
        rng_state_X, rng_state_y = backend.random.split(rng_state, num=2)

        X = self._generate_X(
            rng_state_X=rng_state_X,
            num_data=size[0] + size[1],
            input_shape=input_shape,
            range_X=range_X,
        )

        y = self._generate_y(
            X=X,
            rng_state_y=rng_state_y,
            num_data=size[0] + size[1],
            output_shape=output_shape,
            fun=fun,
            noise_var=noise_var,
        )

        super().__init__(X=X, y=y, train_idcs=backend.arange(size[0]))

    @property
    def fun(self) -> Callable[[backend.Array], backend.Array]:
        return self._fun

    def _generate_X(
        self,
        rng_state_X: backend.random.RNGState,
        num_data: int,
        input_shape: ShapeType,
        range_X: Tuple[float, float],
    ) -> backend.Array:
        """Randomly generate inputs."""
        return backend.random.uniform(
            rng_state=rng_state_X,
            shape=(num_data,) + input_shape,
            minval=range_X[0],
            maxval=range_X[1],
        )

    def _generate_y(
        self,
        X: backend.Array,
        rng_state_y: backend.random.RNGState,
        num_data: int,
        output_shape: ShapeType,
        fun: Callable[[backend.Array], backend.Array],
        noise_var: float,
    ) -> backend.Array:
        """Generate outputs via a ground truth function and some added noise."""
        return fun(X) + backend.sqrt(noise_var) * backend.random.standard_normal(
            rng_state=rng_state_y, shape=(num_data,) + output_shape
        )
