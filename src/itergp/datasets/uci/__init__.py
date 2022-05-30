"""Datasets from the UCI machine learning repository."""

from ._bike_sharing import BikeSharing
from ._kegg_undir import KEGGUndir
from ._parkinsons import Parkinsons
from ._protein import Protein
from ._road_network import RoadNetwork
from ._uci_dataset import UCIDataset

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "UCIDataset",
    "BikeSharing",
    "KEGGUndir",
    "Parkinsons",
    "Protein",
    "RoadNetwork",
]

# Set correct module paths. Corrects links and module paths in documentation.
UCIDataset.__module__ = "itergp.datasets.uci"
BikeSharing.__module__ = "itergp.datasets.uci"
KEGGUndir.__module__ = "itergp.datasets.uci"
Parkinsons.__module__ = "itergp.datasets.uci"
Protein.__module__ = "itergp.datasets.uci"
RoadNetwork.__module__ = "itergp.datasets.uci"
