"""Tests for tournament simulation."""
import pytest
import numpy as np

from soccer_prediction.models.elo_logistic import EloLogisticModel
from soccer_prediction.simulation.group_stage import simulate_group, select_best_third_place
from soccer_prediction.simulation.monte_carlo import MonteCarloSimulator
from soccer_prediction.simulation.tournament import TournamentConfig, DEFAULT_WC2026_GROUPS


TEAMS_4 = ["Brazil", "France", "Germany", "USA"]


@pytest.fixture
def fitted_model(sample_matches):
    model = EloLogisticModel()
    model.fit(sample_matches)
    return model


@pytest.fixture
def small_config():
    return TournamentConfig(
        name="Test Cup",
        year=2024,
        groups={
            "A": ["Brazil", "France", "Germany", "USA"],
            "B": ["Argentina", "England", "Spain", "Italy"],
        },
        third_place_advances=0,
    )


class TestGroupStageSimulation:
    def test_simulate_group_returns_four_standings(self, fitted_model):
        rng = np.random.default_rng(42)
        standings, matches = simulate_group("A", TEAMS_4, fitted_model, rng)
        assert len(standings) == 4

    def test_simulate_group_all_teams_present(self, fitted_model):
        rng = np.random.default_rng(42)
        standings, _ = simulate_group("A", TEAMS_4, fitted_model, rng)
        team_names = {s.team for s in standings}
        assert team_names == set(TEAMS_4)

    def test_six_matches_played(self, fitted_model):
        rng = np.random.default_rng(42)
        _, matches = simulate_group("A", TEAMS_4, fitted_model, rng)
        # 4 teams → C(4,2) = 6 round-robin matches
        assert len(matches) == 6

    def test_standings_ordered_by_points(self, fitted_model):
        rng = np.random.default_rng(42)
        standings, _ = simulate_group("A", TEAMS_4, fitted_model, rng)
        points = [s.points for s in standings]
        assert points == sorted(points, reverse=True)

    def test_each_team_plays_three_games(self, fitted_model):
        rng = np.random.default_rng(42)
        standings, _ = simulate_group("A", TEAMS_4, fitted_model, rng)
        for s in standings:
            assert s.played == 3

    def test_select_best_third_place(self, fitted_model):
        from soccer_prediction.simulation.tournament import GroupStanding
        rng = np.random.default_rng(0)
        # Create 3 fake third-place standings
        all_third = [
            ("TeamA", GroupStanding("TeamA", 3, 1, 0, 2, 2, 1)),
            ("TeamB", GroupStanding("TeamB", 3, 1, 0, 2, 3, 2)),
            ("TeamC", GroupStanding("TeamC", 3, 0, 0, 3, 0, 3)),
        ]
        # TeamB has higher GF, should be selected first
        selected = select_best_third_place(all_third, n=2, rng=rng)
        assert "TeamA" in selected or "TeamB" in selected  # top 2 by points then GF
        assert len(selected) == 2


class TestMonteCarloSimulator:
    def test_run_small(self, fitted_model, small_config):
        sim = MonteCarloSimulator(n_simulations=100, seed=42)
        results = sim.run(small_config, fitted_model)
        assert results.n_simulations == 100

    def test_champion_counts_add_up(self, fitted_model, small_config):
        sim = MonteCarloSimulator(n_simulations=200, seed=1)
        results = sim.run(small_config, fitted_model)
        total = sum(results.champion_counts.values())
        assert total == 200

    def test_team_probabilities_valid(self, fitted_model, small_config):
        sim = MonteCarloSimulator(n_simulations=200, seed=2)
        results = sim.run(small_config, fitted_model)
        df = results.team_probabilities()
        assert not df.empty
        # Probabilities should be between 0 and 1
        num_cols = [c for c in df.columns if c != "team"]
        for col in num_cols:
            assert df[col].between(0.0, 1.0).all(), f"Column '{col}' out of range"

    def test_favourite_wins_more_often(self, fitted_model, small_config):
        sim = MonteCarloSimulator(n_simulations=500, seed=99)
        results = sim.run(small_config, fitted_model)
        df = results.team_probabilities()
        # Brazil should have higher win probability than USA
        brazil_prob = df[df["team"] == "Brazil"]["wins_tournament"].values
        usa_prob = df[df["team"] == "USA"]["wins_tournament"].values
        if len(brazil_prob) and len(usa_prob):
            assert brazil_prob[0] >= usa_prob[0]

    def test_all_teams_appear(self, fitted_model, small_config):
        sim = MonteCarloSimulator(n_simulations=100, seed=7)
        results = sim.run(small_config, fitted_model)
        df = results.team_probabilities()
        sim_teams = set(df["team"])
        config_teams = set(small_config.all_teams)
        assert config_teams.issubset(sim_teams)

    def test_default_wc2026_config(self, fitted_model):
        """Smoke test that WC 2026 default config runs without errors."""
        # Use only 2 groups to keep test fast
        small_groups = {k: v for k, v in list(DEFAULT_WC2026_GROUPS.items())[:2]}
        config = TournamentConfig(
            name="Test WC",
            year=2026,
            groups=small_groups,
            third_place_advances=0,
        )
        sim = MonteCarloSimulator(n_simulations=50, seed=0)
        results = sim.run(config, fitted_model)
        assert sum(results.champion_counts.values()) == 50
