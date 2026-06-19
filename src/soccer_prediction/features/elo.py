"""Elo rating calculator and feature extractor."""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from .base import FeatureExtractor

log = logging.getLogger(__name__)

DEFAULT_ELO = 1500.0
K_FACTOR = 32.0
HOME_ADVANTAGE = 100.0  # Elo points added to home team's rating


class EloCalculator(FeatureExtractor):
    """
    Computes Elo ratings by replaying match history chronologically.

    Features added to the matches DataFrame:
      - elo_home_before, elo_away_before: ratings before the match
      - elo_diff: elo_home_before - elo_away_before (home-adjusted if not neutral)
      - elo_home_after, elo_away_after: updated ratings after the match
    """

    name = "elo"

    def __init__(
        self,
        k_factor: float = K_FACTOR,
        home_advantage: float = HOME_ADVANTAGE,
        initial_ratings: Optional[dict[str, float]] = None,
    ) -> None:
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings: dict[str, float] = dict(initial_ratings or {})
        self._fitted = False

    # ------------------------------------------------------------------
    # FeatureExtractor interface
    # ------------------------------------------------------------------

    def fit(self, matches: pd.DataFrame) -> "EloCalculator":
        """Replay completed matches to compute final Elo ratings."""
        if matches.empty:
            self._fitted = True
            return self

        completed = matches[matches["home_goals"].notna()].copy()
        if "match_date" in completed.columns:
            completed = completed.sort_values("match_date")

        for _, row in completed.iterrows():
            self._update(
                row["home_team"],
                row["away_team"],
                int(row["home_goals"]),
                int(row["away_goals"]),
                bool(row.get("is_neutral", True)),
            )

        log.info("Elo fitted on %d completed matches. %d teams tracked.", len(completed), len(self.ratings))
        self._fitted = True
        return self

    def transform(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Add Elo columns. For completed matches, also adds elo_after columns."""
        out = matches.copy()
        ratings_snapshot = dict(self.ratings)

        elo_home_before, elo_away_before, elo_diff_col = [], [], []
        elo_home_after, elo_away_after = [], []

        temp = dict(ratings_snapshot)

        if "match_date" in out.columns:
            order = out.sort_values("match_date").index
        else:
            order = out.index

        result_map: dict = {}
        for idx in order:
            row = out.loc[idx]
            h, a = row["home_team"], row["away_team"]
            r_h = temp.get(h, DEFAULT_ELO)
            r_a = temp.get(a, DEFAULT_ELO)
            is_neutral = bool(row.get("is_neutral", True))
            adj = 0 if is_neutral else self.home_advantage
            diff = (r_h + adj) - r_a

            r_h_after, r_a_after = r_h, r_a
            if pd.notna(row.get("home_goals")):
                r_h_after, r_a_after = self._compute_new_ratings(
                    r_h, r_a, int(row["home_goals"]), int(row["away_goals"]), is_neutral
                )
                temp[h] = r_h_after
                temp[a] = r_a_after

            result_map[idx] = (r_h, r_a, diff, r_h_after, r_a_after)

        for idx in out.index:
            r_h, r_a, diff, r_h_after, r_a_after = result_map[idx]
            elo_home_before.append(r_h)
            elo_away_before.append(r_a)
            elo_diff_col.append(diff)
            elo_home_after.append(r_h_after)
            elo_away_after.append(r_a_after)

        out["elo_home_before"] = elo_home_before
        out["elo_away_before"] = elo_away_before
        out["elo_diff"] = elo_diff_col
        out["elo_home_after"] = elo_home_after
        out["elo_away_after"] = elo_away_after
        return out

    def get_match_features(self, home_team: str, away_team: str, **context) -> dict:
        is_neutral = context.get("is_neutral", True)
        r_h = self.ratings.get(home_team, DEFAULT_ELO)
        r_a = self.ratings.get(away_team, DEFAULT_ELO)
        adj = 0.0 if is_neutral else self.home_advantage
        return {
            "elo_home": r_h,
            "elo_away": r_a,
            "elo_diff": (r_h + adj) - r_a,
        }

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_rating(self, team: str) -> float:
        return self.ratings.get(team, DEFAULT_ELO)

    def get_all_ratings(self) -> dict[str, float]:
        return dict(self.ratings)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def _actual_score(self, goals_a: int, goals_b: int) -> float:
        if goals_a > goals_b:
            return 1.0
        if goals_a < goals_b:
            return 0.0
        return 0.5

    def _compute_new_ratings(
        self,
        r_home: float,
        r_away: float,
        home_goals: int,
        away_goals: int,
        is_neutral: bool,
    ) -> tuple[float, float]:
        adj = 0.0 if is_neutral else self.home_advantage
        e_home = self._expected_score(r_home + adj, r_away)
        s_home = self._actual_score(home_goals, away_goals)
        delta = self.k_factor * (s_home - e_home)
        return r_home + delta, r_away - delta

    def _update(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        is_neutral: bool,
    ) -> None:
        r_h = self.ratings.get(home_team, DEFAULT_ELO)
        r_a = self.ratings.get(away_team, DEFAULT_ELO)
        r_h_new, r_a_new = self._compute_new_ratings(r_h, r_a, home_goals, away_goals, is_neutral)
        self.ratings[home_team] = r_h_new
        self.ratings[away_team] = r_a_new
