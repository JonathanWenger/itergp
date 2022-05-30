"""Protein dataset from the UCI machine learning repository."""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
from probnum import backend

from ._uci_dataset import UCIDataset


class Parkinsons(UCIDataset):
    """Parkinsons Telemonitoring dataset (5,875 × 21). [1]_

    This dataset is composed of a range of biomedical voice measurements from 42 people
    with early-stage Parkinson's disease recruited to a six-month trial of a
    telemonitoring device for remote symptom progression monitoring. The recordings were
    automatically captured in the patient's homes. The original study used a range of
    linear and nonlinear regression methods to predict the clinician's Parkinson's
    disease symptom score on the UPDRS scale.

    Source: https://archive.ics.uci.edu/ml/datasets/parkinsons+telemonitoring

    References
    ----------
    .. [1] A Tsanas, MA Little, PE McSharry, LO Ramig (2009) 'Accurate telemonitoring of
           Parkinson's disease progression by non-invasive speech tests', IEEE
           Transactions on Biomedical Engineering
    """

    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/"

    def __init__(
        self, dir: Optional[str] = "data/uci/parkinsons", overwrite: bool = False
    ):
        super().__init__(dir, overwrite)

    @staticmethod
    def _download() -> backend.Array:
        # Read data
        df = pd.read_csv(Parkinsons.URL + "parkinsons_updrs.data")
        df.drop(["motor_UPDRS"], axis=1)

        # Move column to predict
        column_to_move = df.pop("total_UPDRS")
        df.insert(0, "total_UPDRS", column_to_move)

        return backend.asarray(df)

    @staticmethod
    def _preprocess(
        raw_data: backend.Array,
    ) -> Tuple[backend.Array, backend.Array, backend.Array, backend.Array]:

        # Preprocess
        X = raw_data[:, 1::]
        y = raw_data[:, 0]

        # Transform outputs
        y = y - backend.mean(y, axis=0)

        # Normalize features
        X = (X - backend.mean(X, axis=0)) / backend.std(X, axis=0)

        # Select train-test split
        train_idcs = UCIDataset._get_train_idcs(
            rng_state=backend.random.rng_state(4589), num_data=X.shape[0]
        )

        return X, y, train_idcs
