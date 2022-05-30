"""Protein dataset from the UCI machine learning repository."""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
from probnum import backend

from ._uci_dataset import UCIDataset


class KEGGUndir(UCIDataset):
    """KEGG Metabolic pathways (Undirected) dataset (63,608 × 26).

    KEGG Metabolic pathways modelled as a graph. A variety of network features were
    computed using Cytoscape. [1]_

    Source: https://archive.ics.uci.edu/ml/datasets/KEGG+Metabolic+Reaction+Network+(Undirected)

    References
    ----------
    .. [1] Shannon,P., Markiel,A., Ozier,O., Baliga,N.S., Wang,J.T.,Ramage,D., Amin,N.,
           Schwikowski,B. and Ideker,T. (2003) Cytoscape: a software environment for
           integrated models of biomolecular interaction networks. Genome Res., 13
    """

    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00221/"

    def __init__(self, dir: Optional[str] = "data/uci/keggu", overwrite: bool = False):
        super().__init__(dir, overwrite)

    @staticmethod
    def _download() -> backend.Array:
        # Read data
        df = pd.read_csv(
            KEGGUndir.URL + "Reaction%20Network%20(Undirected).data",
            index_col=0,
            header=None,
        )
        df.drop(df[df[4] == "?"].index, inplace=True)
        df[4] = df[4].astype(float)
        df.drop(df[df[21] > 1].index, inplace=True)
        df.drop(columns=[10], inplace=True)

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
            rng_state=backend.random.rng_state(48879), num_data=X.shape[0]
        )

        return X, y, train_idcs
