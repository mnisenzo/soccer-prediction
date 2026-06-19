"""Monte Carlo tournament simulator for the FIFA World Cup."""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

from ..models.base import SoccerModel
from .group_stage import select_best_third_place, simulate_group
from .tournament import SimulatedMatch, TournamentConfig

log = logging.getLogger(__name__)


class SimulationResults:
    """
    Stores aggregated results from N Monte Carlo tournament simulations.

    Attributes:
        n_simulations: number of trials run
        team_stage_counts: {team: {stage: count_reached}}
        group_winner_counts: {group: {team: count}}
        match_results: {(home, away): {'H': n, 'D': n, 'A': n, 'total_goals': [...]}}
    """

    def __init__(self, n_simulations: int) -> None:
        self.n_simulations = n_simulations
        self.team_stage_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.group_winner_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.third_place_advance_counts: dict[str, int] = defaultdict(int)
        self.match_results: dict[tuple, dict] = defaultdict(
            lambda: {"H": 0, "D": 0, "A": 0, "total_goals": []}
        )
        self.champion_counts: dict[str, int] = defaultdict(int)

    def record_group_result(
        self,
        group_name: str,
        standings,
        matches: list[SimulatedMatch],
        third_place_advancing: set[str],
    ) -> None:
        for rank, standing in enumerate(standings):
            team = standing.team
            self.team_stage_counts[team]["group"] += 1
            if rank == 0:
                self.group_winner_counts[group_name][team] += 1
                self.team_stage_counts[team]["group_winner"] += 1
            if rank <= 1 or team in third_place_advancing:
                self.team_stage_counts[team]["advances_from_group"] += 1
            if team in third_place_advancing and rank == 2:
                self.third_place_advance_counts[team] += 1

        for m in matches:
            self._record_match(m)

    def record_knockout_result(self, stage: str, match: SimulatedMatch) -> None:
        self._record_match(match)

    def record_stage_entry(self, stage: str, teams: list[str]) -> None:
        """Record all teams that entered a knockout round."""
        for t in teams:
            self.team_stage_counts[t][f"reaches_{stage}"] += 1

    def _record_match(self, m: SimulatedMatch) -> None:
        key = (m.home_team, m.away_team)
        r = self.match_results[key]
        r[m.result] += 1
        r["total_goals"].append(m.home_goals + m.away_goals)

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def team_probabilities(self) -> pd.DataFrame:
        """Return a DataFrame of per-team stage probabilities."""
        n = self.n_simulations
        rows = []
        for team, stage_counts in self.team_stage_counts.items():
            row = {"team": team}
            for stage, cnt in stage_counts.items():
                row[stage] = cnt / n
            row["wins_tournament"] = self.champion_counts.get(team, 0) / n
            rows.append(row)
        df = pd.DataFrame(rows).fillna(0.0)
        if "wins_tournament" not in df.columns:
            df["wins_tournament"] = 0.0
        return df.sort_values("wins_tournament", ascending=False).reset_index(drop=True)

    def group_winner_probabilities(self) -> pd.DataFrame:
        n = self.n_simulations
        rows = []
        for group, counts in self.group_winner_counts.items():
            for team, cnt in counts.items():
                rows.append({"group": group, "team": team, "prob_group_winner": cnt / n})
        return pd.DataFrame(rows).sort_values(["group", "prob_group_winner"], ascending=[True, False])

    def match_probabilities(self) -> pd.DataFrame:
        n = self.n_simulations
        rows = []
        for (home, away), counts in self.match_results.items():
            total = counts["H"] + counts["D"] + counts["A"]
            if total == 0:
                continue
            goals = counts["total_goals"]
            rows.append({
                "home_team": home,
                "away_team": away,
                "home_win_prob": counts["H"] / total,
                "draw_prob": counts["D"] / total,
                "away_win_prob": counts["A"] / total,
                "avg_total_goals": float(np.mean(goals)) if goals else 0.0,
                "prob_over_2_5": sum(1 for g in goals if g > 2.5) / len(goals) if goals else 0.0,
                "prob_btts": 0.0,  # requires score tracking — set by Monte Carlo below
            })
        return pd.DataFrame(rows)


class MonteCarloSimulator:
    """
    Simulates the FIFA World Cup N times to produce stage-reach probabilities.

    Usage::

        sim = MonteCarloSimulator(n_simulations=10_000)
        results = sim.run(config, model)
        df = results.team_probabilities()
    """

    def __init__(self, n_simulations: int = 10_000, seed: Optional[int] = None) -> None:
        self.n_simulations = n_simulations
        self.seed = seed

    def run(
        self,
        config: TournamentConfig,
        model: SoccerModel,
        progress_callback=None,
    ) -> SimulationResults:
        """
        Run N tournament simulations.

        Args:
            config: TournamentConfig describing group assignments and format.
            model: fitted SoccerModel to generate match predictions.
            progress_callback: optional callable(i, n) for progress updates.

        Returns:
            SimulationResults aggregating all N trials.
        """
        rng = np.random.default_rng(self.seed)
        results = SimulationResults(self.n_simulations)

        log.info("Starting Monte Carlo: %d simulations for %s %d",
                 self.n_simulations, config.name, config.year)

        for i in range(self.n_simulations):
            self._run_one(config, model, rng, results)
            if progress_callback and i % max(1, self.n_simulations // 100) == 0:
                progress_callback(i + 1, self.n_simulations)

        log.info("Simulation complete.")
        return results

    # ------------------------------------------------------------------

    def _run_one(
        self,
        config: TournamentConfig,
        model: SoccerModel,
        rng: np.random.Generator,
        results: SimulationResults,
    ) -> None:
        all_third_place: list[tuple[str, object]] = []
        group_runners: dict[str, list[str]] = {}  # group → [1st, 2nd]

        # --- Group stage ---
        for group_name, teams in config.groups.items():
            standings, matches = simulate_group(group_name, teams, model, rng)
            third_place_team = standings[2].team if len(standings) > 2 else None
            if third_place_team:
                all_third_place.append((third_place_team, standings[2]))

            advancing_third: set[str] = set()
            # Don't know yet which 3rd-place teams advance; resolve after all groups
            results.record_group_result(group_name, standings, matches, advancing_third)
            group_runners[group_name] = [s.team for s in standings[:2]]

        # Determine best 3rd-place teams
        n_third = config.third_place_advances
        best_third = select_best_third_place(all_third_place, n_third, rng)

        # Retroactively update advances_from_group for best 3rd-place
        for team in best_third:
            results.team_stage_counts[team]["advances_from_group"] += 1
            results.third_place_advance_counts[team] += 1

        # --- Build knockout bracket ---
        # Seed order: group winners first (by group alphabetical), then runners-up, then best 3rd
        group_names = sorted(config.groups.keys())
        bracket: list[str] = (
            [group_runners[g][0] for g in group_names]  # 12 group winners
            + [group_runners[g][1] for g in group_names]  # 12 runners-up
            + best_third  # 8 best 3rd-place
        )

        # Shuffle bracket within pairs to avoid systematic bias (neutral draw)
        rng.shuffle(bracket)

        # --- Knockout rounds ---
        remaining = list(bracket)
        for stage in config.knockout_rounds:
            results.record_stage_entry(stage, remaining)
            remaining = self._play_knockout_round(stage, remaining, model, rng, results)

        if remaining:
            results.champion_counts[remaining[0]] += 1

    def _play_knockout_round(
        self,
        stage: str,
        teams: list[str],
        model: SoccerModel,
        rng: np.random.Generator,
        results: SimulationResults,
    ) -> list[str]:
        winners = []
        for idx in range(0, len(teams), 2):
            home = teams[idx]
            away = teams[idx + 1]
            match = self._simulate_knockout_match(home, away, model, rng, stage)
            results.record_knockout_result(stage, match)
            if match.winner:
                winners.append(match.winner)
        return winners

    def _simulate_knockout_match(
        self,
        home: str,
        away: str,
        model: SoccerModel,
        rng: np.random.Generator,
        stage: str,
    ) -> SimulatedMatch:
        """Simulate a knockout match (draws resolved by penalties)."""
        pred = model.predict_match(home, away, is_neutral=True)
        hg, ag = pred.sample_outcome(rng)

        if hg == ag:
            # Penalty shootout: roughly 50/50 for each team
            if rng.random() < 0.5:
                hg += 1  # fictional extra goal to indicate penalty winner
            else:
                ag += 1

        return SimulatedMatch(
            home_team=home,
            away_team=away,
            home_goals=hg,
            away_goals=ag,
            stage=stage,
        )
