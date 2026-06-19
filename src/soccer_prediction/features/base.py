"""Abstract base class for feature extractors."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class FeatureExtractor(ABC):
    """Computes a set of features from a matches DataFrame and attaches them to the output."""

    name: str  # unique identifier for this extractor

    @abstractmethod
    def fit(self, matches: pd.DataFrame) -> "FeatureExtractor":
        """Learn parameters from historical matches."""

    @abstractmethod
    def transform(self, matches: pd.DataFrame) -> pd.DataFrame:
        """
        Add feature columns to `matches` and return the augmented DataFrame.
        Does not modify `matches` in place.
        """

    def fit_transform(self, matches: pd.DataFrame) -> pd.DataFrame:
        return self.fit(matches).transform(matches)

    @abstractmethod
    def get_match_features(self, home_team: str, away_team: str, **context) -> dict:
        """Return features for a single prospective match as a flat dict."""
