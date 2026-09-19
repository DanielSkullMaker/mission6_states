"""Pipeline orchestrator package (Фаза 6): FursovPipeline + именованные стадии."""

from subspace_conjugacy.pipeline.fursov_pipeline import FursovPipeline
from subspace_conjugacy.pipeline.stages import STAGE_REGISTRY

__all__ = [
    "FursovPipeline",
    "STAGE_REGISTRY",
]
