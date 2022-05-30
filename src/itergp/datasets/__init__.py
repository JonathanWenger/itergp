"""Datasets for experiments."""
from . import uci
from ._dataset import Dataset
from ._synthetic_dataset import SyntheticDataset

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "Dataset",
    "SyntheticDataset",
]

# Set correct module paths. Corrects links and module paths in documentation.
Dataset.__module__ = "itergp.datasets"
SyntheticDataset.__module__ = "itergp.datasets"
