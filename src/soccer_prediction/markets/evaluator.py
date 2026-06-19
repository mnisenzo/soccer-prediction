"""Market evaluator: converts simulation results into market probabilities."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..models.base import SoccerModel
from ..simulation.monte_carlo import SimulationResults
from .base import MarketDefinition, MarketResult

log = logging.getLogger(__name__)

# Maps stage names as used in market params → internal stage keys
_STAGE_KEY_MAP = {
    "round_of_32": "reaches_round_of_32",
    "round_of_16": "reaches_round_of_16",
    "quarterfinal": "reaches_quarterfinal",
    "semifinal": "reaches_semifinal",
    "final": "reaches_final",
    "winner": "wins_tournament",
    "tournament": "wins_tournament",
}


class MarketEvaluator:
    """
    Evaluates MarketDefinitions against simulation results and/or model predictions.

    For tournament-level markets: uses SimulationResults aggregates.
    For match-level markets: uses the SoccerModel directly.
    """

    def __init__(
        self,
        sim_results: Optional[SimulationResults] = None,
        model: Optional[SoccerModel] = None,
    ) -> None:
        self.sim_results = sim_results
        self.model = model

    def evaluate(self, market: MarketDefinition) -> MarketResult:
        """Evaluate a single market and return its probability."""
        try:
            prob = self._dispatch(market)
        except Exception as exc:
            log.warning("Failed to evaluate market '%s': %s", market.id, exc)
            prob = float("nan")

        return MarketResult(
            market=market,
            model_probability=prob,
            market_price=market.market_price,
        )

    def evaluate_all(self, markets: list[MarketDefinition]) -> list[MarketResult]:
        return [self.evaluate(m) for m in markets]

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def _dispatch(self, m: MarketDefinition) -> float:
        cat = m.category
        if cat == "tournament_winner":
            return self._tournament_winner(m)
        if cat == "reaches_stage":
            return self._reaches_stage(m)
        if cat == "advances_from_group":
            return self._advances_from_group(m)
        if cat == "group_winner":
            return self._group_winner(m)
        if cat == "match_outcome":
            return self._match_outcome(m)
        if cat == "over_under_goals":
            return self._over_under_goals(m)
        if cat == "both_teams_score":
            return self._both_teams_score(m)
        if cat == "clean_sheet":
            return self._clean_sheet(m)
        if cat == "correct_score":
            return self._correct_score(m)
        if cat == "anytime_winner":
            return self._tournament_winner(m)
        raise ValueError(f"Unsupported market category: '{cat}'")

    # ------------------------------------------------------------------
    # Tournament-level (require SimulationResults)
    # ------------------------------------------------------------------

    def _require_sim(self) -> SimulationResults:
        if self.sim_results is None:
            raise RuntimeError("SimulationResults required but not provided.")
        return self.sim_results

    def _tournament_winner(self, m: MarketDefinition) -> float:
        sim = self._require_sim()
        team = m.params["team"]
        n = sim.n_simulations
        return sim.champion_counts.get(team, 0) / n

    def _reaches_stage(self, m: MarketDefinition) -> float:
        sim = self._require_sim()
        team = m.params["team"]
        stage = m.params["stage"]
        key = _STAGE_KEY_MAP.get(stage, f"reaches_{stage}")
        n = sim.n_simulations
        return sim.team_stage_counts.get(team, {}).get(key, 0) / n

    def _advances_from_group(self, m: MarketDefinition) -> float:
        sim = self._require_sim()
        team = m.params["team"]
        n = sim.n_simulations
        return sim.team_stage_counts.get(team, {}).get("advances_from_group", 0) / n

    def _group_winner(self, m: MarketDefinition) -> float:
        sim = self._require_sim()
        team = m.params["team"]
        group = m.params.get("group", "")
        n = sim.n_simulations
        if group:
            return sim.group_winner_counts.get(group, {}).get(team, 0) / n
        # Search all groups
        for g_counts in sim.group_winner_counts.values():
            if team in g_counts:
                return g_counts[team] / n
        return 0.0

    # ------------------------------------------------------------------
    # Match-level (require Model)
    # ------------------------------------------------------------------

    def _require_model(self) -> SoccerModel:
        if self.model is None:
            raise RuntimeError("SoccerModel required but not provided.")
        return self.model

    def _get_match_pred(self, m: MarketDefinition):
        model = self._require_model()
        home = m.params["home_team"]
        away = m.params["away_team"]
        is_neutral = m.params.get("is_neutral", True)
        return model.predict_match(home, away, is_neutral=is_neutral)

    def _match_outcome(self, m: MarketDefinition) -> float:
        pred = self._get_match_pred(m)
        outcome = m.params["outcome"].upper()
        if outcome in ("HOME", "H", "1"):
            return pred.home_win_prob
        if outcome in ("DRAW", "D", "X"):
            return pred.draw_prob
        if outcome in ("AWAY", "A", "2"):
            return pred.away_win_prob
        raise ValueError(f"Unknown outcome: '{outcome}'. Use H, D, or A.")

    def _over_under_goals(self, m: MarketDefinition) -> float:
        pred = self._get_match_pred(m)
        threshold = float(m.params["threshold"])
        direction = m.params.get("direction", "over").lower()

        if pred.score_matrix is not None:
            total_goals = np.arange(pred.score_matrix.shape[0])[:, None] + \
                          np.arange(pred.score_matrix.shape[1])[None, :]
            if direction == "over":
                prob = float(pred.score_matrix[total_goals > threshold].sum())
            else:
                prob = float(pred.score_matrix[total_goals <= threshold].sum())
        else:
            # Poisson approximation using expected goals
            from scipy.stats import poisson
            mu = (pred.home_goals_exp or 1.3) + (pred.away_goals_exp or 1.1)
            if direction == "over":
                prob = 1.0 - poisson.cdf(int(threshold), mu)
            else:
                prob = poisson.cdf(int(threshold), mu)
        return float(np.clip(prob, 0.0, 1.0))

    def _both_teams_score(self, m: MarketDefinition) -> float:
        pred = self._get_match_pred(m)
        if pred.score_matrix is not None:
            mat = pred.score_matrix
            # BTTS: home > 0 AND away > 0, i.e. exclude row 0 and column 0
            btts = float(mat[1:, 1:].sum())
        else:
            from scipy.stats import poisson
            mu_h = pred.home_goals_exp or 1.3
            mu_a = pred.away_goals_exp or 1.1
            p_h_scores = 1.0 - poisson.pmf(0, mu_h)
            p_a_scores = 1.0 - poisson.pmf(0, mu_a)
            btts = p_h_scores * p_a_scores
        return float(np.clip(btts, 0.0, 1.0))

    def _clean_sheet(self, m: MarketDefinition) -> float:
        pred = self._get_match_pred(m)
        team = m.params["team"]
        home_team = m.params["home_team"]
        away_team = m.params["away_team"]
        # Clean sheet for team = opponent scores 0
        if team == home_team:
            # Away scores 0
            if pred.score_matrix is not None:
                prob = float(pred.score_matrix[:, 0].sum())
            else:
                from scipy.stats import poisson
                prob = poisson.pmf(0, pred.away_goals_exp or 1.1)
        else:
            # Home scores 0
            if pred.score_matrix is not None:
                prob = float(pred.score_matrix[0, :].sum())
            else:
                from scipy.stats import poisson
                prob = poisson.pmf(0, pred.home_goals_exp or 1.3)
        return float(np.clip(prob, 0.0, 1.0))

    def _correct_score(self, m: MarketDefinition) -> float:
        pred = self._get_match_pred(m)
        hg = int(m.params["home_goals"])
        ag = int(m.params["away_goals"])
        if pred.score_matrix is not None:
            if hg < pred.score_matrix.shape[0] and ag < pred.score_matrix.shape[1]:
                return float(pred.score_matrix[hg, ag])
            return 0.0
        from scipy.stats import poisson
        mu_h = pred.home_goals_exp or 1.3
        mu_a = pred.away_goals_exp or 1.1
        return float(poisson.pmf(hg, mu_h) * poisson.pmf(ag, mu_a))
