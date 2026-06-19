"""Elo-based logistic model for 3-way match outcome prediction."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from ..features.elo import EloCalculator
from .base import MatchPrediction, SoccerModel

log = logging.getLogger(__name__)


class EloLogisticModel(SoccerModel):
    """
    Predicts win/draw/loss probabilities using Elo rating differences.

    Mathematical model:
      E_home = 1 / (1 + 10^(-(elo_home - elo_away) / scale))
      P(draw) = draw_base * (1 - |2*E_home - 1|)   # peaks when evenly matched
      P(home_win) = E_home * (1 - P(draw))
      P(away_win) = (1-E_home) * (1 - P(draw))

    This is a closed-form formula; no ML fitting required for the 3-way split.
    The Elo ratings themselves are updated from match history.
    """

    name = "elo_logistic"

    def __init__(
        self,
        k_factor: float = 32.0,
        home_advantage: float = 100.0,
        elo_scale: float = 400.0,
        draw_base: float = 0.28,
        initial_ratings: Optional[dict[str, float]] = None,
    ) -> None:
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.elo_scale = elo_scale
        self.draw_base = draw_base
        self.elo = EloCalculator(
            k_factor=k_factor,
            home_advantage=home_advantage,
            initial_ratings=initial_ratings,
        )
        self._fitted = False

    # ------------------------------------------------------------------

    def fit(self, matches: pd.DataFrame) -> "EloLogisticModel":
        """Replay match history to compute current Elo ratings for all teams."""
        self.elo.fit(matches)
        self._fitted = True
        log.info("EloLogisticModel fitted on %d matches.", len(matches[matches["home_goals"].notna()]))
        return self

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        is_neutral: bool = True,
        **context,
    ) -> MatchPrediction:
        r_home = self.elo.get_rating(home_team)
        r_away = self.elo.get_rating(away_team)
        adj = 0.0 if is_neutral else self.home_advantage

        e_home = 1.0 / (1.0 + 10.0 ** (-((r_home + adj) - r_away) / self.elo_scale))
        draw_prob = self.draw_base * (1.0 - abs(2.0 * e_home - 1.0))
        home_win = e_home * (1.0 - draw_prob)
        away_win = (1.0 - e_home) * (1.0 - draw_prob)

        return MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            home_win_prob=home_win,
            draw_prob=draw_prob,
            away_win_prob=away_win,
        )

    def get_ratings(self) -> dict[str, float]:
        return self.elo.get_all_ratings()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        log.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "EloLogisticModel":
        model = joblib.load(Path(path))
        log.info("Model loaded from %s", path)
        return model
