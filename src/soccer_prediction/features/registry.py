"""Registry of available feature extractors."""
from __future__ import annotations

from typing import Optional

from .base import FeatureExtractor
from .elo import EloCalculator
from .form import RecentFormExtractor
from .pipeline import FeaturePipeline

_REGISTRY: dict[str, type[FeatureExtractor]] = {
    "elo": EloCalculator,
    "recent_form": RecentFormExtractor,
}


def register_extractor(name: str, cls: type[FeatureExtractor]) -> None:
    """Register a custom FeatureExtractor so the pipeline can find it by name."""
    _REGISTRY[name] = cls


def get_extractor(name: str, **kwargs) -> FeatureExtractor:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown feature extractor: '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def list_extractors() -> list[str]:
    return list(_REGISTRY.keys())


def build_default_pipeline() -> FeaturePipeline:
    """Return a pipeline with all default extractors."""
    return FeaturePipeline([EloCalculator(), RecentFormExtractor(window=5)])
