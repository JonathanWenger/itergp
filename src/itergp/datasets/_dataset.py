"""Base class for datasets."""
from __future__ import annotations

from collections import namedtuple
import os
import pathlib
from typing import Tuple

import numpy as np
from probnum import backend
from probnum.backend.random import RNGState
from probnum.backend.typing import ShapeType

TrainData = namedtuple("TrainData", ["X", "y"])
TestData = namedtuple("TestData", ["X", "y"])


class Dataset:
    """Classification / Regression dataset.

    Parameters
    ----------
    X
        Inputs / features.
    y
        Outputs / labels.
    train_idcs
        Indices of the training data.
    """

    def __init__(self, X: backend.Array, y: backend.Array, train_idcs: backend.Array):
        self._X = X
        self._y = y
        self._train_idcs = train_idcs
        test_mask = backend.ones(X.shape[0], dtype=backend.bool)
        test_mask[train_idcs] = False
        self._test_idcs = backend.arange(X.shape[0])[test_mask]

    def __len__(self) -> int:
        return self._y.shape[0]

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}: "
            f"input_shape={self.input_shape}, "
            f"n_train={self.train.y.shape[0]}, "
            f"n_test={self.test.y.shape[0]}>"
        )

    @property
    def train(self) -> TrainData:
        """Training data."""
        return TrainData(
            X=self._X[self._train_idcs, ...], y=self._y[self._train_idcs, ...]
        )

    @property
    def test(self) -> TestData:
        "Test data."
        return TestData(
            X=self._X[self._test_idcs, ...], y=self._y[self._test_idcs, ...]
        )

    @property
    def input_shape(self) -> ShapeType:
        """Input shape of the data."""
        return self._X.shape[1:]

    @property
    def output_shape(self) -> ShapeType:
        """Output shape of the data."""
        return self._y.shape[1:]

    @classmethod
    def from_disk(cls, dir: str) -> "Dataset":
        return cls(*cls._load(dir=dir))

    @classmethod
    def _load(
        cls,
        dir: str = "data",
    ) -> Tuple[backend.Array, backend.Array, backend.Array]:
        dir = pathlib.Path(dir)

        file = dir / (cls.__name__.lower() + ".npz")
        dataset = np.load(file=file)

        X = backend.asarray(dataset["X"])
        y = backend.asarray(dataset["y"])
        train_idcs = backend.asarray(dataset["train_idcs"])

        return X, y, train_idcs

    def save(self, dir: str = "data", overwrite=False) -> None:
        """Save dataset to disk.

        Parameters
        ----------
        dir
            Directory to save data to.
        overwrite
            Whether to overwrite existing data.
        """
        dir = pathlib.Path(dir)
        dir.mkdir(parents=True, exist_ok=True)

        file = dir / (self.__class__.__name__.lower() + ".npz")

        if not os.path.isfile(file) or overwrite:
            np.savez_compressed(
                file=file,
                X=np.asarray(self._X),
                y=np.asarray(self._y),
                train_idcs=np.asarray(self._train_idcs),
            )

    def resample(self, rng_state: RNGState) -> "Dataset":
        """Resample the training and test set from the entire data set.

        Randomly selects new datapoints for the training and test set of the same sizes
        as the original dataset.

        Parameters
        ----------
        rng_state
            Random number generator state.
        """
        train_idcs = backend.random.permutation(rng_state, len(self))[
            0 : len(self._train_idcs)
        ]

        return Dataset(
            X=self._X,
            y=self._y,
            train_idcs=train_idcs,
        )
