"""Road network dataset from the UCI machine learning repository."""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
from probnum import backend

from ._uci_dataset import UCIDataset


class RoadNetwork(UCIDataset):
    """3D Road Network (North Jutland, Denmark) (434,874 × 3).

    Dataset of longitude, latitude and altitude values of a road network in
    North Jutland, Denmark (covering a region of 185x135 km2). Elevation values where
    extracted from a publicly available massive Laser Scan Point Cloud for Denmark. This
    3D road network was eventually used for benchmarking various fuel and CO2 estimation
    algorithms.

    Source: https://archive.ics.uci.edu/ml/datasets/3D+Road+Network+(North+Jutland,+Denmark)
    """

    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00246/"

    def __init__(self, dir: Optional[str] = "data/uci/3droad", overwrite: bool = False):
        super().__init__(dir, overwrite)

    @staticmethod
    def _download() -> backend.Array:
        # Read data
        df = pd.read_csv(
            RoadNetwork.URL + "3D_spatial_network.txt",
            header=None,
            names=["OSM_ID", "longitude", "latitude", "altitude"],
        )

        return backend.asarray(df)

    @staticmethod
    def _preprocess(
        raw_data: backend.Array,
    ) -> Tuple[backend.Array, backend.Array, backend.Array]:

        # Preprocess
        X = raw_data[:, 1:-1]
        y = raw_data[:, -1]

        # Transform outputs
        y = y - backend.mean(y, axis=0)

        # Normalize features
        X = (X - backend.mean(X, axis=0)) / backend.std(X, axis=0)

        # Select train-test split
        train_idcs = UCIDataset._get_train_idcs(
            rng_state=backend.random.rng_state(126), num_data=X.shape[0]
        )

        return X, y, train_idcs
