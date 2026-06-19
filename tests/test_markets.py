"""Tests for market definitions and evaluation."""
import pytest
import math

from soccer_prediction.markets.base import MarketDefinition, MarketResult, MARKET_CATEGORIES
from soccer_prediction.markets.evaluator import MarketEvaluator
from soccer_prediction.markets.registry import MarketRegistry
from soccer_prediction.models.elo_logistic import EloLogisticModel
from soccer_prediction.models.poisson import PoissonGoalsModel
from soccer_prediction.simulation.monte_carlo import MonteCarloSimulator
from soccer_prediction.simulation.tournament import TournamentConfig


@pytest.fixture
def elo_model(sample_matches):
    m = EloLogisticModel()
    m.fit(sample_matches)
    return m


@pytest.fixture
def poisson_model(sample_matches):
    m = PoissonGoalsModel()
    m.fit(sample_matches)
    return m


@pytest.fixture
def sim_results(elo_model):
    groups = {
        "A": ["Brazil", "France", "Germany", "USA"],
        "B": ["Argentina", "England", "Spain", "Italy"],
    }
    config = TournamentConfig(name="Test", year=2024, groups=groups, third_place_advances=0)
    sim = MonteCarloSimulator(n_simulations=500, seed=42)
    return sim.run(config, elo_model)


class TestMarketDefinition:
    def test_valid_category(self):
        m = MarketDefinition("test", "Test", "tournament_winner", {"team": "Brazil"})
        assert m.id == "test"

    def test_invalid_category_raises(self):
        with pytest.raises(ValueError, match="Unknown market category"):
            MarketDefinition("x", "X", "invalid_category", {})

    def test_to_dict(self):
        m = MarketDefinition("id", "Name", "tournament_winner", {"team": "Brazil"}, market_price=0.15)
        d = m.to_dict()
        assert d["id"] == "id"
        assert d["market_price"] == 0.15


class TestMarketResult:
    def test_edge_computed_correctly(self):
        m = MarketDefinition("test", "Test", "tournament_winner", {"team": "X"}, market_price=0.20)
        result = MarketResult(m, model_probability=0.25, market_price=0.20)
        assert abs(result.edge - 0.05) < 1e-6

    def test_edge_none_when_no_market_price(self):
        m = MarketDefinition("test", "Test", "tournament_winner", {"team": "X"})
        result = MarketResult(m, model_probability=0.25)
        assert result.edge is None


class TestMarketEvaluator:
    def test_tournament_winner(self, sim_results, elo_model):
        evaluator = MarketEvaluator(sim_results=sim_results, model=elo_model)
        m = MarketDefinition("bra_win", "Brazil wins", "tournament_winner", {"team": "Brazil"})
        result = evaluator.evaluate(m)
        assert 0.0 <= result.model_probability <= 1.0

    def test_advances_from_group(self, sim_results, elo_model):
        evaluator = MarketEvaluator(sim_results=sim_results, model=elo_model)
        m = MarketDefinition("bra_adv", "Brazil advances", "advances_from_group", {"team": "Brazil"})
        result = evaluator.evaluate(m)
        assert result.model_probability > 0.5, "Brazil should usually advance"

    def test_group_winner(self, sim_results, elo_model):
        evaluator = MarketEvaluator(sim_results=sim_results, model=elo_model)
        m = MarketDefinition("bra_gw", "Brazil wins group A", "group_winner",
                             {"team": "Brazil", "group": "A"})
        result = evaluator.evaluate(m)
        assert 0.0 <= result.model_probability <= 1.0

    def test_match_outcome_home_win(self, elo_model):
        evaluator = MarketEvaluator(model=elo_model)
        m = MarketDefinition("bra_hw", "Brazil beats USA", "match_outcome",
                             {"home_team": "Brazil", "away_team": "USA", "outcome": "H", "is_neutral": True})
        result = evaluator.evaluate(m)
        assert result.model_probability > 0.4, "Brazil should frequently beat USA"

    def test_over_under_goals_poisson(self, poisson_model):
        evaluator = MarketEvaluator(model=poisson_model)
        m_over = MarketDefinition("over", "Over 2.5", "over_under_goals",
                                  {"home_team": "Brazil", "away_team": "USA",
                                   "threshold": 2.5, "direction": "over", "is_neutral": True})
        m_under = MarketDefinition("under", "Under 2.5", "over_under_goals",
                                   {"home_team": "Brazil", "away_team": "USA",
                                    "threshold": 2.5, "direction": "under", "is_neutral": True})
        r_over = evaluator.evaluate(m_over)
        r_under = evaluator.evaluate(m_under)
        # Over + Under should sum to ~1
        assert abs(r_over.model_probability + r_under.model_probability - 1.0) < 0.01

    def test_both_teams_score(self, poisson_model):
        evaluator = MarketEvaluator(model=poisson_model)
        m = MarketDefinition("btts", "BTTS", "both_teams_score",
                             {"home_team": "France", "away_team": "Germany", "is_neutral": True})
        result = evaluator.evaluate(m)
        assert 0.0 <= result.model_probability <= 1.0

    def test_clean_sheet(self, poisson_model):
        evaluator = MarketEvaluator(model=poisson_model)
        m = MarketDefinition("cs", "Brazil clean sheet", "clean_sheet",
                             {"team": "Brazil", "home_team": "Brazil",
                              "away_team": "USA", "is_neutral": True})
        result = evaluator.evaluate(m)
        assert 0.0 < result.model_probability < 1.0

    def test_correct_score(self, poisson_model):
        evaluator = MarketEvaluator(model=poisson_model)
        m = MarketDefinition("cs10", "Brazil 1-0 USA", "correct_score",
                             {"home_team": "Brazil", "away_team": "USA",
                              "home_goals": 1, "away_goals": 0, "is_neutral": True})
        result = evaluator.evaluate(m)
        assert 0.0 < result.model_probability < 0.25

    def test_reaches_stage_semifinal(self, sim_results, elo_model):
        evaluator = MarketEvaluator(sim_results=sim_results, model=elo_model)
        m = MarketDefinition("bra_sf", "Brazil reaches SF", "reaches_stage",
                             {"team": "Brazil", "stage": "semifinal"})
        result = evaluator.evaluate(m)
        assert 0.0 <= result.model_probability <= 1.0

    def test_missing_sim_raises_for_tournament_market(self, elo_model):
        evaluator = MarketEvaluator(model=elo_model)  # no sim_results
        m = MarketDefinition("x", "X", "tournament_winner", {"team": "Brazil"})
        result = evaluator.evaluate(m)
        assert math.isnan(result.model_probability)


class TestMarketRegistry:
    def test_load_yaml(self, tmp_path):
        yaml_content = """
markets:
  - id: test_market
    name: "Test Market"
    category: tournament_winner
    params:
      team: Brazil
    market_price: 0.15
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content)
        reg = MarketRegistry()
        n = reg.load_yaml(yaml_file)
        assert n == 1
        m = reg.get("test_market")
        assert m is not None
        assert m.market_price == 0.15

    def test_load_missing_file_returns_zero(self, tmp_path):
        reg = MarketRegistry()
        n = reg.load_yaml(tmp_path / "nonexistent.yaml")
        assert n == 0

    def test_update_price(self, tmp_path):
        yaml_content = """
markets:
  - id: m1
    name: "M1"
    category: tournament_winner
    params:
      team: France
"""
        (tmp_path / "m.yaml").write_text(yaml_content)
        reg = MarketRegistry()
        reg.load_yaml(tmp_path / "m.yaml")
        reg.update_price("m1", 0.22)
        assert reg.get("m1").market_price == 0.22
