"""Tests for feature extractors."""
import pytest
import pandas as pd

from soccer_prediction.features.elo import EloCalculator, DEFAULT_ELO
from soccer_prediction.features.form import RecentFormExtractor
from soccer_prediction.features.pipeline import FeaturePipeline
from soccer_prediction.features.registry import build_default_pipeline, list_extractors


class TestEloCalculator:
    def test_fit_updates_ratings(self, sample_matches):
        elo = EloCalculator()
        elo.fit(sample_matches)
        assert "Brazil" in elo.ratings
        assert "USA" in elo.ratings

    def test_winner_gains_elo(self, sample_matches):
        elo = EloCalculator()
        elo.fit(sample_matches)
        # Brazil beat Argentina 2-1 in first match; should have higher rating after
        assert elo.get_rating("Brazil") != DEFAULT_ELO

    def test_unknown_team_returns_default(self):
        elo = EloCalculator()
        assert elo.get_rating("Atlantis FC") == DEFAULT_ELO

    def test_transform_adds_columns(self, sample_matches):
        elo = EloCalculator()
        result = elo.fit_transform(sample_matches)
        for col in ("elo_home_before", "elo_away_before", "elo_diff"):
            assert col in result.columns, f"Missing column: {col}"

    def test_get_match_features(self, sample_matches):
        elo = EloCalculator()
        elo.fit(sample_matches)
        feats = elo.get_match_features("Brazil", "France", is_neutral=True)
        assert "elo_home" in feats
        assert "elo_diff" in feats
        assert feats["elo_home"] > 1000

    def test_home_advantage_applied(self, sample_matches):
        elo = EloCalculator(home_advantage=100)
        elo.fit(sample_matches)
        feats_neutral = elo.get_match_features("Germany", "Spain", is_neutral=True)
        feats_home = elo.get_match_features("Germany", "Spain", is_neutral=False)
        assert feats_home["elo_diff"] > feats_neutral["elo_diff"]

    def test_initial_ratings_respected(self):
        elo = EloCalculator(initial_ratings={"Brazil": 2000, "USA": 1600})
        assert elo.get_rating("Brazil") == 2000
        assert elo.get_rating("USA") == 1600


class TestRecentFormExtractor:
    def test_fit_builds_history(self, sample_matches):
        form = RecentFormExtractor(window=5)
        form.fit(sample_matches)
        assert len(form._team_history) > 0

    def test_transform_adds_form_columns(self, sample_matches):
        form = RecentFormExtractor()
        result = form.fit_transform(sample_matches)
        for col in ("home_form_wins", "home_form_gf", "away_form_ga"):
            assert col in result.columns, f"Missing column: {col}"

    def test_no_history_returns_zeros(self):
        form = RecentFormExtractor()
        form.fit(pd.DataFrame())  # empty
        feats = form.get_match_features("Atlantis", "Wakanda")
        assert feats["home_form_wins"] == 0
        assert feats["away_form_gf"] == 0.0

    def test_get_match_features(self, sample_matches):
        form = RecentFormExtractor(window=5)
        form.fit(sample_matches)
        feats = form.get_match_features("Brazil", "Argentina")
        assert "home_form_points" in feats
        assert 0.0 <= feats["home_form_points"] <= 3.0


class TestFeaturePipeline:
    def test_pipeline_combines_features(self, sample_matches):
        pipeline = build_default_pipeline()
        pipeline.fit(sample_matches)
        feats = pipeline.get_match_features("Brazil", "France")
        # Should have both Elo and form features
        assert "elo_diff" in feats
        assert "home_form_points" in feats

    def test_registry_lists_extractors(self):
        names = list_extractors()
        assert "elo" in names
        assert "recent_form" in names
