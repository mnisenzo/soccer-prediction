"""Registry of available soccer models."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import SoccerModel
from .elo_logistic import EloLogisticModel
from .poisson import PoissonGoalsModel

_REGISTRY: dict[str, type[SoccerModel]] = {
    EloLogisticModel.name: EloLogisticModel,
    PoissonGoalsModel.name: PoissonGoalsModel,
}


def register_model(cls: type[SoccerModel]) -> type[SoccerModel]:
    """Decorator to register a custom model class."""
    _REGISTRY[cls.name] = cls
    return cls


def get_model_class(name: str) -> type[SoccerModel]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model: '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]


def list_models() -> list[str]:
    return list(_REGISTRY.keys())


class ModelRegistry:
    """
    Manages instantiated, fitted model instances.

    Keeps one fitted instance per model name so the app can reuse them
    without re-fitting on every request.
    """

    def __init__(self) -> None:
        self._models: dict[str, SoccerModel] = {}

    def fit_and_register(self, name: str, matches, **kwargs) -> SoccerModel:
        """Instantiate, fit, and store a model by name."""
        cls = get_model_class(name)
        model = cls(**kwargs)
        model.fit(matches)
        self._models[name] = model
        return model

    def get(self, name: str) -> Optional[SoccerModel]:
        return self._models.get(name)

    def get_or_fit(self, name: str, matches, **kwargs) -> SoccerModel:
        if name not in self._models:
            return self.fit_and_register(name, matches, **kwargs)
        return self._models[name]

    def save_all(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, model in self._models.items():
            model.save(directory / f"{name}.joblib")

    def load(self, name: str, path: Path) -> SoccerModel:
        cls = get_model_class(name)
        model = cls.load(path)
        self._models[name] = model
        return model

    def available(self) -> list[str]:
        return list(self._models.keys())
