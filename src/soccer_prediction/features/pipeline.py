"""Feature pipeline: orchestrates multiple extractors in sequence."""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from .base import FeatureExtractor

log = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Runs a list of FeatureExtractors in order.

    Usage::

        pipeline = FeaturePipeline([EloCalculator(), RecentFormExtractor()])
        pipeline.fit(historical_matches)
        features_df = pipeline.transform(fixture_df)
        match_feats = pipeline.get_match_features("Brazil", "France")
    """

    def __init__(self, extractors: Optional[list[FeatureExtractor]] = None) -> None:
        self.extractors: list[FeatureExtractor] = extractors or []

    def add(self, extractor: FeatureExtractor) -> "FeaturePipeline":
        self.extractors.append(extractor)
        return self

    def fit(self, matches: pd.DataFrame) -> "FeaturePipeline":
        for ext in self.extractors:
            log.info("Fitting feature extractor: %s", ext.name)
            ext.fit(matches)
        return self

    def transform(self, matches: pd.DataFrame) -> pd.DataFrame:
        df = matches.copy()
        for ext in self.extractors:
            df = ext.transform(df)
        return df

    def fit_transform(self, matches: pd.DataFrame) -> pd.DataFrame:
        return self.fit(matches).transform(matches)

    def get_match_features(self, home_team: str, away_team: str, **context) -> dict:
        features: dict = {}
        for ext in self.extractors:
            features.update(ext.get_match_features(home_team, away_team, **context))
        return features

    def get_extractor(self, name: str) -> Optional[FeatureExtractor]:
        for ext in self.extractors:
            if ext.name == name:
                return ext
        return None
