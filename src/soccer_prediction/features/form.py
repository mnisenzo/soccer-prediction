"""Recent-form feature extractor."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .base import FeatureExtractor

log = logging.getLogger(__name__)


class RecentFormExtractor(FeatureExtractor):
    """
    Computes rolling form statistics over the last N completed matches for each team.

    Features per match row (prefixed home_ / away_):
      - {side}_form_wins, _draws, _losses: counts in last N matches
      - {side}_form_gf, _ga, _gd: goals for/against/difference per game
      - {side}_form_points: average points per game (3/1/0)
    """

    name = "recent_form"

    def __init__(self, window: int = 5) -> None:
        self.window = window
        self._team_history: dict[str, list[dict]] = {}
        self._fitted = False

    def fit(self, matches: pd.DataFrame) -> "RecentFormExtractor":
        """Build team-level match history from completed matches in chronological order."""
        self._team_history = {}
        if matches.empty or "home_goals" not in matches.columns:
            self._fitted = True
            return self
        completed = matches[matches["home_goals"].notna()].copy()
        if "match_date" in completed.columns:
            completed = completed.sort_values("match_date")

        for _, row in completed.iterrows():
            h, a = row["home_team"], row["away_team"]
            hg, ag = int(row["home_goals"]), int(row["away_goals"])
            date = row.get("match_date")

            for team, gf, ga in [(h, hg, ag), (a, ag, hg)]:
                self._team_history.setdefault(team, []).append({
                    "date": date,
                    "gf": gf,
                    "ga": ga,
                    "result": "W" if gf > ga else ("D" if gf == ga else "L"),
                })

        self._fitted = True
        log.info("RecentForm fitted. %d teams with history.", len(self._team_history))
        return self

    def transform(self, matches: pd.DataFrame) -> pd.DataFrame:
        out = matches.copy()
        for side in ("home", "away"):
            col = f"{side}_team"
            stats = out[col].apply(lambda t: self._get_form_stats(t))
            stats_df = pd.DataFrame(stats.tolist(), index=out.index)
            stats_df.columns = [f"{side}_{c}" for c in stats_df.columns]
            out = pd.concat([out, stats_df], axis=1)
        return out

    def get_match_features(self, home_team: str, away_team: str, **context) -> dict:
        h_stats = self._get_form_stats(home_team)
        a_stats = self._get_form_stats(away_team)
        return {
            **{f"home_{k}": v for k, v in h_stats.items()},
            **{f"away_{k}": v for k, v in a_stats.items()},
        }

    # ------------------------------------------------------------------

    def _get_form_stats(self, team: str) -> dict:
        history = self._team_history.get(team, [])[-self.window:]
        if not history:
            return {
                "form_wins": 0, "form_draws": 0, "form_losses": 0,
                "form_gf": 0.0, "form_ga": 0.0, "form_gd": 0.0,
                "form_points": 0.0,
            }
        wins = sum(1 for g in history if g["result"] == "W")
        draws = sum(1 for g in history if g["result"] == "D")
        losses = sum(1 for g in history if g["result"] == "L")
        gf = np.mean([g["gf"] for g in history])
        ga = np.mean([g["ga"] for g in history])
        points = (3 * wins + draws) / len(history)
        return {
            "form_wins": wins,
            "form_draws": draws,
            "form_losses": losses,
            "form_gf": round(gf, 3),
            "form_ga": round(ga, 3),
            "form_gd": round(gf - ga, 3),
            "form_points": round(points, 3),
        }
