"""Base class for datasets."""

from __future__ import annotations

import abc
import os
from typing import Optional, Tuple

from probnum import backend
from probnum.backend.random import RNGState

from .._dataset import Dataset


class UCIDataset(Dataset, abc.ABC):
    """UCI dataset.

    Parameters
    ----------
    dir
        Directory where data is retrieved from or saved to. If ``None``, data is not
        saved to file.
    overwrite
        Whether to overwrite any potentially saved data on disk.
    """

    def __init__(self, dir: Optional[str] = "data/uci", overwrite: bool = False):

        file = dir + "/" + self.__class__.__name__.lower() + ".npz"
        if os.path.isfile(file) and not overwrite:
            # Load data from file
            X, y, train_idcs = self._load(dir=dir)
        else:
            # Download data from the web
            raw_data = self._download()
            X, y, train_idcs = self._preprocess(raw_data)

        super().__init__(X=X, y=y, train_idcs=train_idcs)

        # Save to disk if data does not yet exist on disk
        if dir is not None:
            self.save(dir=dir, overwrite=overwrite)

    @staticmethod
    def _download() -> backend.Array:
        """Download data from the UCI repository."""
        raise NotImplementedError

    @staticmethod
    def _preprocess(
        raw_data: backend.Array,
    ) -> Tuple[backend.Array, backend.Array, backend.Array]:
        """Preprocess, normalize and split into train and test set."""
        raise NotImplementedError

    @staticmethod
    def _get_train_idcs(
        rng_state: RNGState, num_data: int, train_fraction: float = 0.9
    ) -> backend.Array:
        """Sample a set of training indices."""
        train_set_size = int(num_data * train_fraction)
        return backend.random.permutation(rng_state, num_data)[0:train_set_size]
