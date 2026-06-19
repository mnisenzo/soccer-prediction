"""Abstract model interface and MatchPrediction dataclass."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class MatchPrediction:
    """Output of a single match prediction."""

    home_team: str
    away_team: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    home_goals_exp: Optional[float] = None
    away_goals_exp: Optional[float] = None
    # [max_goals+1, max_goals+1] matrix; (i, j) = P(home=i, away=j)
    score_matrix: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        total = self.home_win_prob + self.draw_prob + self.away_win_prob
        if abs(total - 1.0) > 1e-4:
            # Auto-normalise to absorb floating-point drift
            self.home_win_prob /= total
            self.draw_prob /= total
            self.away_win_prob /= total

    def sample_outcome(self, rng: Optional[np.random.Generator] = None) -> tuple[int, int]:
        """
        Sample a (home_goals, away_goals) scoreline.
        Uses score_matrix if available; falls back to outcome-only sampling.
        """
        rng = rng or np.random.default_rng()
        if self.score_matrix is not None:
            flat = self.score_matrix.ravel()
            idx = rng.choice(len(flat), p=flat / flat.sum())
            n = self.score_matrix.shape[0]
            return divmod(idx, n)

        # Fallback: sample outcome then assign a plausible score
        outcome = rng.choice(["H", "D", "A"], p=[self.home_win_prob, self.draw_prob, self.away_win_prob])
        mu_h = self.home_goals_exp or 1.3
        mu_a = self.away_goals_exp or 1.1
        for _ in range(100):
            h = int(rng.poisson(mu_h))
            a = int(rng.poisson(mu_a))
            if outcome == "H" and h > a:
                return h, a
            if outcome == "D" and h == a:
                return h, a
            if outcome == "A" and a > h:
                return h, a
        # Deterministic fallback
        if outcome == "H":
            return 1, 0
        if outcome == "D":
            return 1, 1
        return 0, 1

    def to_dict(self) -> dict:
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_win_prob": round(self.home_win_prob, 4),
            "draw_prob": round(self.draw_prob, 4),
            "away_win_prob": round(self.away_win_prob, 4),
            "home_goals_exp": round(self.home_goals_exp, 3) if self.home_goals_exp else None,
            "away_goals_exp": round(self.away_goals_exp, 3) if self.away_goals_exp else None,
        }


class SoccerModel(ABC):
    """
    Abstract base for all soccer prediction models.

    To add a new model:
    1. Subclass SoccerModel.
    2. Implement fit(), predict_match(), save(), load().
    3. Register it in models/registry.py.
    """

    name: str  # human-readable identifier, must be unique

    @abstractmethod
    def fit(self, matches: pd.DataFrame) -> "SoccerModel":
        """
        Train the model on historical completed matches.

        Args:
            matches: DataFrame with columns home_team, away_team, home_goals,
                     away_goals, match_date, is_neutral, stage.
        """

    @abstractmethod
    def predict_match(
        self,
        home_team: str,
        away_team: str,
        is_neutral: bool = True,
        **context,
    ) -> MatchPrediction:
        """
        Predict the outcome of a single match.

        Args:
            home_team: Name of the home/first team.
            away_team: Name of the away/second team.
            is_neutral: If True, no home-field advantage is applied.
            **context: Optional extra info (stage, date, elo overrides, etc.).

        Returns:
            MatchPrediction with win/draw/loss probabilities.
        """

    def predict_tournament(self, fixtures: pd.DataFrame) -> list[MatchPrediction]:
        """
        Predict all matches in a fixtures DataFrame.

        Fixtures must have columns: home_team, away_team, is_neutral.
        """
        preds = []
        for _, row in fixtures.iterrows():
            preds.append(
                self.predict_match(
                    row["home_team"],
                    row["away_team"],
                    is_neutral=bool(row.get("is_neutral", True)),
                )
            )
        return preds

    @abstractmethod
    def save(self, path: Path) -> None:
        """Serialize the model to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "SoccerModel":
        """Deserialize a saved model from disk."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
