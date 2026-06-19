"""Poisson goals model for soccer match prediction."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson

from .base import MatchPrediction, SoccerModel

log = logging.getLogger(__name__)

MAX_GOALS = 10  # Upper bound for score matrix computation


class PoissonGoalsModel(SoccerModel):
    """
    Predicts match outcomes using independent Poisson distributions for each team's goals.

    Expected goals per team:
        lambda_home = attack[home] * defense[away] * avg_goals
        lambda_away = attack[away] * defense[home] * avg_goals
        (home_advantage multiplier applied to lambda_home when not neutral)

    Attack/defense parameters are normalised team averages relative to the league mean.
    The joint score probability matrix P(home=i, away=j) is derived from the Poisson PMFs.
    """

    name = "poisson_goals"

    def __init__(
        self,
        home_advantage: float = 1.15,
        min_matches: int = 3,
    ) -> None:
        self.home_advantage = home_advantage
        self.min_matches = min_matches

        self.avg_goals: float = 1.3
        self.attack: dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self._fitted = False

    # ------------------------------------------------------------------

    def fit(self, matches: pd.DataFrame) -> "PoissonGoalsModel":
        completed = matches[matches["home_goals"].notna()].copy()
        if completed.empty:
            log.warning("No completed matches to fit Poisson model.")
            self._fitted = True
            return self

        total_goals = int(completed["home_goals"].sum() + completed["away_goals"].sum())
        total_matches = len(completed)
        self.avg_goals = total_goals / (2 * total_matches) if total_matches > 0 else 1.3

        teams = set(completed["home_team"]) | set(completed["away_team"])
        goals_scored: dict[str, list[int]] = {t: [] for t in teams}
        goals_conceded: dict[str, list[int]] = {t: [] for t in teams}

        for _, row in completed.iterrows():
            h, a = row["home_team"], row["away_team"]
            hg, ag = int(row["home_goals"]), int(row["away_goals"])
            goals_scored[h].append(hg)
            goals_scored[a].append(ag)
            goals_conceded[h].append(ag)
            goals_conceded[a].append(hg)

        for team in teams:
            n = len(goals_scored[team])
            if n >= self.min_matches:
                self.attack[team] = np.mean(goals_scored[team]) / self.avg_goals
                self.defense[team] = np.mean(goals_conceded[team]) / self.avg_goals
            else:
                # Shrink towards league average for teams with few matches
                self.attack[team] = 1.0
                self.defense[team] = 1.0

        self._fitted = True
        log.info(
            "PoissonGoalsModel fitted on %d matches. avg_goals=%.3f, %d teams.",
            total_matches, self.avg_goals, len(teams),
        )
        return self

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        is_neutral: bool = True,
        **context,
    ) -> MatchPrediction:
        mu_h = self._lambda(home_team, away_team)
        mu_a = self._lambda(away_team, home_team)
        if not is_neutral:
            mu_h *= self.home_advantage

        score_matrix = self._score_matrix(mu_h, mu_a)
        home_win = float(np.sum(np.tril(score_matrix, -1)))
        draw = float(np.sum(np.diag(score_matrix)))
        away_win = float(np.sum(np.triu(score_matrix, 1)))

        return MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            home_win_prob=home_win,
            draw_prob=draw,
            away_win_prob=away_win,
            home_goals_exp=round(mu_h, 3),
            away_goals_exp=round(mu_a, 3),
            score_matrix=score_matrix,
        )

    # ------------------------------------------------------------------

    def _lambda(self, attack_team: str, defense_team: str) -> float:
        att = self.attack.get(attack_team, 1.0)
        def_ = self.defense.get(defense_team, 1.0)
        return max(att * def_ * self.avg_goals, 0.1)

    def _score_matrix(self, mu_h: float, mu_a: float) -> np.ndarray:
        """P(home=i, away=j) for i, j in 0..MAX_GOALS."""
        h_pmf = np.array([poisson.pmf(i, mu_h) for i in range(MAX_GOALS + 1)])
        a_pmf = np.array([poisson.pmf(j, mu_a) for j in range(MAX_GOALS + 1)])
        matrix = np.outer(h_pmf, a_pmf)
        # Normalise to account for truncation at MAX_GOALS
        matrix /= matrix.sum()
        return matrix

    def score_probabilities(self, home_team: str, away_team: str, is_neutral: bool = True) -> pd.DataFrame:
        """Return a DataFrame of the most likely scorelines."""
        pred = self.predict_match(home_team, away_team, is_neutral)
        if pred.score_matrix is None:
            return pd.DataFrame()
        rows = []
        for i in range(MAX_GOALS + 1):
            for j in range(MAX_GOALS + 1):
                rows.append({"home_goals": i, "away_goals": j, "probability": pred.score_matrix[i, j]})
        df = pd.DataFrame(rows).sort_values("probability", ascending=False)
        return df[df["probability"] > 0.001].reset_index(drop=True)

    def get_attack_defense_table(self) -> pd.DataFrame:
        """Return team attack/defense parameters as a DataFrame."""
        rows = [
            {"team": t, "attack": self.attack.get(t, 1.0), "defense": self.defense.get(t, 1.0)}
            for t in sorted(set(self.attack) | set(self.defense))
        ]
        return pd.DataFrame(rows).sort_values("attack", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        log.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "PoissonGoalsModel":
        model = joblib.load(Path(path))
        log.info("Model loaded from %s", path)
        return model
