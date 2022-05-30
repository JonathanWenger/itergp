"""Bike sharing dataset from the UCI machine learning repository."""

from __future__ import annotations

from io import BytesIO
from typing import Optional, Tuple
from zipfile import ZipFile

import pandas as pd
from probnum import backend
import requests

from ._uci_dataset import UCIDataset


class BikeSharing(UCIDataset):
    """Bike sharing dataset (17,379 × 16). [1]_

    This dataset contains the hourly (and daily) count of rental bikes between years
    2011 and 2012 of the Capital bikeshare system with the corresponding weather and
    seasonal information.

    Source: https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset

    References
    ----------
    .. [1] Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors
           and background knowledge", Progress in Artificial Intelligence (2013): pp.
           1-15, Springer Berlin Heidelberg, doi:10.1007/s13748-013-0040-3.
    """

    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/"

    def __init__(self, dir: Optional[str] = "data/uci/bike", overwrite: bool = False):
        super().__init__(dir, overwrite)

    @staticmethod
    def _download() -> backend.Array:
        # Download and unzip archive
        r = requests.get(BikeSharing.URL + "Bike-Sharing-Dataset.zip")
        files = ZipFile(BytesIO(r.content))

        # Read data for the hourly count
        df = pd.read_csv(files.open("hour.csv"))

        # Convert dates to numeric
        df["dteday"] = pd.to_datetime(df["dteday"]).astype(int)

        return backend.asarray(df)

    @staticmethod
    def _preprocess(
        raw_data: backend.Array,
    ) -> Tuple[backend.Array, backend.Array, backend.Array, backend.Array]:

        # Preprocess
        X = raw_data[:, 0:-1]
        y = raw_data[:, -1]

        # Transform outputs
        y = backend.log(y)
        y = y - backend.mean(y, axis=0)

        # Normalize features
        X = (X - backend.mean(X, axis=0)) / backend.std(X, axis=0)

        # Select train-test split
        train_idcs = UCIDataset._get_train_idcs(
            rng_state=backend.random.rng_state(2494), num_data=X.shape[0]
        )

        return X, y, train_idcs
