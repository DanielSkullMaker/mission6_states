"""Algorithms package for canonical clustering and legacy notebook compatibility."""

from subspace_conjugacy.algorithms.global_pair import GlobalMinCosinePairFinder
from subspace_conjugacy.algorithms.reference_centers import ReferenceCenterBuilder
from subspace_conjugacy.algorithms.subclass_seed import CosineSecondVectorAttacher
from subspace_conjugacy.algorithms.subclass_growth import ConjugacyClusterGrowth
from subspace_conjugacy.algorithms.fursov_clusterer import FursovClusterer, SubspaceClusterer
from subspace_conjugacy.algorithms.subclass_export import (
    flatten_subspace_bases,
    unflatten_subspace_bases,
    export_clusterer_bases,
    export_all_classes,
)

__all__ = [
    "GlobalMinCosinePairFinder",
    "ReferenceCenterBuilder",
    "CosineSecondVectorAttacher",
    "ConjugacyClusterGrowth",
    "FursovClusterer",
    "SubspaceClusterer",
    "flatten_subspace_bases",
    "unflatten_subspace_bases",
    "export_clusterer_bases",
    "export_all_classes",
]
