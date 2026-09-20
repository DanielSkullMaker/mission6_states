"""Algorithms package for canonical clustering and legacy notebook compatibility."""

from subspace_conjugacy.algorithms.global_pair import GlobalMinCosinePairFinder
from subspace_conjugacy.algorithms.reference_centers import ReferenceCenterBuilder
from subspace_conjugacy.algorithms.reference_filter import LinearDependencyFilter
from subspace_conjugacy.algorithms.informativeness_filter import (
    DEFAULT_BRIGHTNESS_THRESHOLD,
    DEFAULT_MIN_FRACTION_OF_MEAN,
    LowInformativenessFilter,
)
from subspace_conjugacy.algorithms.subclass_seed import CosineSecondVectorAttacher
from subspace_conjugacy.algorithms.subclass_growth import ConjugacyClusterGrowth
from subspace_conjugacy.algorithms.fursov_clusterer import FursovClusterer, SubspaceClusterer
from subspace_conjugacy.algorithms.subclass_export import (
    equalize_subspace_bases,
    flatten_subspace_bases,
    unflatten_subspace_bases,
    export_clusterer_bases,
    export_all_classes,
)

__all__ = [
    "GlobalMinCosinePairFinder",
    "ReferenceCenterBuilder",
    "LinearDependencyFilter",
    "LowInformativenessFilter",
    "DEFAULT_BRIGHTNESS_THRESHOLD",
    "DEFAULT_MIN_FRACTION_OF_MEAN",
    "CosineSecondVectorAttacher",
    "ConjugacyClusterGrowth",
    "FursovClusterer",
    "SubspaceClusterer",
    "equalize_subspace_bases",
    "flatten_subspace_bases",
    "unflatten_subspace_bases",
    "export_clusterer_bases",
    "export_all_classes",
]
