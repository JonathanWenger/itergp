"""Protein dataset from the UCI machine learning repository."""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
from probnum import backend

from ._uci_dataset import UCIDataset


class Protein(UCIDataset):
    """Protein dataset (45,730 × 9).

    This is a data set of Physicochemical Properties of Protein Tertiary Structure. The
    data set is taken from CASP 5-9. There are 45730 decoys and size varying from 0 to
    21 armstrong.

    Source: https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure
    """

    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/"

    def __init__(
        self, dir: Optional[str] = "data/uci/protein", overwrite: bool = False
    ):
        super().__init__(dir, overwrite)

    @staticmethod
    def _download() -> backend.Array:
        # Read data
        df = pd.read_csv(Protein.URL + "CASP.csv")

        return backend.asarray(df)

    @staticmethod
    def _preprocess(
        raw_data: backend.Array,
    ) -> Tuple[backend.Array, backend.Array, backend.Array]:

        # Preprocess
        X = raw_data[:, 1::]
        y = raw_data[:, 0]

        # Transform outputs
        y = backend.log(y + 1)
        y = y - backend.mean(y, axis=0)

        # Normalize features
        X = (X - backend.mean(X, axis=0)) / backend.std(X, axis=0)

        # Select train-test split
        train_idcs = UCIDataset._get_train_idcs(
            rng_state=backend.random.rng_state(876), num_data=X.shape[0]
        )

        return X, y, train_idcs
