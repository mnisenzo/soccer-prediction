"""Tests for model interface and implementations."""
import math
from pathlib import Path

import pytest

from soccer_prediction.models.base import MatchPrediction
from soccer_prediction.models.elo_logistic import EloLogisticModel
from soccer_prediction.models.poisson import PoissonGoalsModel


class TestMatchPrediction:
    def test_probabilities_sum_to_one(self):
        pred = MatchPrediction("A", "B", 0.5, 0.25, 0.25)
        assert abs(pred.home_win_prob + pred.draw_prob + pred.away_win_prob - 1.0) < 1e-6

    def test_auto_normalise(self):
        # Slightly off-sum is auto-corrected
        pred = MatchPrediction("A", "B", 0.50, 0.26, 0.25)
        total = pred.home_win_prob + pred.draw_prob + pred.away_win_prob
        assert abs(total - 1.0) < 1e-6

    def test_sample_outcome_home_win(self):
        pred = MatchPrediction("A", "B", 1.0, 0.0, 0.0)
        for _ in range(10):
            h, a = pred.sample_outcome()
            assert h > a

    def test_sample_outcome_away_win(self):
        pred = MatchPrediction("A", "B", 0.0, 0.0, 1.0)
        for _ in range(10):
            h, a = pred.sample_outcome()
            assert a > h

    def test_to_dict(self):
        pred = MatchPrediction("A", "B", 0.5, 0.25, 0.25)
        d = pred.to_dict()
        assert d["home_team"] == "A"
        assert d["home_win_prob"] == pytest.approx(0.5, abs=0.001)


class TestEloLogisticModel:
    def test_fit_and_predict(self, sample_matches):
        model = EloLogisticModel()
        model.fit(sample_matches)
        pred = model.predict_match("Brazil", "USA", is_neutral=True)
        assert pred.home_win_prob > pred.away_win_prob, "Brazil should be favoured over USA"

    def test_probs_sum_to_one(self, sample_matches):
        model = EloLogisticModel()
        model.fit(sample_matches)
        pred = model.predict_match("France", "England")
        total = pred.home_win_prob + pred.draw_prob + pred.away_win_prob
        assert abs(total - 1.0) < 1e-6

    def test_even_match_high_draw(self, sample_matches):
        """When teams are evenly matched, draw probability should be near its maximum."""
        model = EloLogisticModel()
        model.fit(sample_matches)
        # Force equal Elo by using the same team
        model.elo.ratings["TeamA"] = 1500
        model.elo.ratings["TeamB"] = 1500
        pred = model.predict_match("TeamA", "TeamB")
        assert pred.draw_prob > 0.25, "Even teams should have elevated draw probability"

    def test_home_advantage(self, sample_matches):
        model = EloLogisticModel()
        model.fit(sample_matches)
        pred_neutral = model.predict_match("Germany", "Spain", is_neutral=True)
        pred_home = model.predict_match("Germany", "Spain", is_neutral=False)
        assert pred_home.home_win_prob > pred_neutral.home_win_prob

    def test_unknown_team_uses_default_elo(self, sample_matches):
        model = EloLogisticModel()
        model.fit(sample_matches)
        pred = model.predict_match("UnknownTeam", "Brazil")
        assert 0 < pred.away_win_prob < 1

    def test_save_load(self, sample_matches, tmp_path):
        model = EloLogisticModel()
        model.fit(sample_matches)
        path = tmp_path / "model.joblib"
        model.save(path)
        loaded = EloLogisticModel.load(path)
        pred_orig = model.predict_match("Brazil", "France")
        pred_load = loaded.predict_match("Brazil", "France")
        assert abs(pred_orig.home_win_prob - pred_load.home_win_prob) < 1e-9


class TestPoissonGoalsModel:
    def test_fit_and_predict(self, sample_matches):
        model = PoissonGoalsModel()
        model.fit(sample_matches)
        pred = model.predict_match("Brazil", "USA", is_neutral=True)
        assert pred.home_goals_exp is not None
        assert pred.away_goals_exp is not None
        assert pred.score_matrix is not None

    def test_score_matrix_sums_to_one(self, sample_matches):
        model = PoissonGoalsModel()
        model.fit(sample_matches)
        pred = model.predict_match("France", "Italy")
        assert abs(pred.score_matrix.sum() - 1.0) < 1e-4

    def test_probs_sum_to_one(self, sample_matches):
        model = PoissonGoalsModel()
        model.fit(sample_matches)
        pred = model.predict_match("Spain", "Germany")
        total = pred.home_win_prob + pred.draw_prob + pred.away_win_prob
        assert abs(total - 1.0) < 1e-4

    def test_favourite_wins_more(self, sample_matches):
        model = PoissonGoalsModel()
        model.fit(sample_matches)
        # Brazil scored many goals, so should be favoured
        pred = model.predict_match("Brazil", "USA")
        assert pred.home_win_prob > 0.3

    def test_score_probabilities(self, sample_matches):
        model = PoissonGoalsModel()
        model.fit(sample_matches)
        df = model.score_probabilities("France", "Germany")
        assert not df.empty
        assert df["probability"].sum() > 0.9

    def test_save_load(self, sample_matches, tmp_path):
        model = PoissonGoalsModel()
        model.fit(sample_matches)
        path = tmp_path / "poisson.joblib"
        model.save(path)
        loaded = PoissonGoalsModel.load(path)
        pred_orig = model.predict_match("Brazil", "England")
        pred_load = loaded.predict_match("Brazil", "England")
        assert abs(pred_orig.home_win_prob - pred_load.home_win_prob) < 1e-9
