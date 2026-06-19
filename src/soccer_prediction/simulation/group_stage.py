"""Group stage simulation logic with WC-style tiebreakers."""
from __future__ import annotations

import random
from itertools import combinations
from typing import Optional

import numpy as np

from ..models.base import SoccerModel
from .tournament import GroupStanding, SimulatedMatch, TournamentConfig


def simulate_group(
    group_name: str,
    teams: list[str],
    model: SoccerModel,
    rng: np.random.Generator,
) -> tuple[list[GroupStanding], list[SimulatedMatch]]:
    """
    Simulate a round-robin group stage for the given teams.

    Returns:
        standings: sorted list of GroupStanding (1st to last)
        matches: list of SimulatedMatch played
    """
    standings = {t: GroupStanding(team=t) for t in teams}
    matches: list[SimulatedMatch] = []
    # H2H records for tiebreaker: (team_a, team_b) → (pts_a, pts_b, gd_a)
    h2h: dict[frozenset, dict] = {}

    for home, away in combinations(teams, 2):
        pred = model.predict_match(home, away, is_neutral=True)
        hg, ag = pred.sample_outcome(rng)

        match = SimulatedMatch(
            home_team=home,
            away_team=away,
            home_goals=hg,
            away_goals=ag,
            stage="group",
            group_name=group_name,
        )
        matches.append(match)

        # Update standings
        for team, gf, ga in [(home, hg, ag), (away, ag, hg)]:
            s = standings[team]
            s.played += 1
            s.gf += gf
            s.ga += ga
            if gf > ga:
                s.wins += 1
            elif gf == ga:
                s.draws += 1
            else:
                s.losses += 1

        # H2H
        key = frozenset({home, away})
        if key not in h2h:
            h2h[key] = {home: {"pts": 0, "gd": 0}, away: {"pts": 0, "gd": 0}}
        if hg > ag:
            h2h[key][home]["pts"] += 3
        elif ag > hg:
            h2h[key][away]["pts"] += 3
        else:
            h2h[key][home]["pts"] += 1
            h2h[key][away]["pts"] += 1
        h2h[key][home]["gd"] += hg - ag
        h2h[key][away]["gd"] += ag - hg

    sorted_standings = _sort_standings(list(standings.values()), h2h, rng)
    return sorted_standings, matches


def _sort_standings(
    standings: list[GroupStanding],
    h2h: dict,
    rng: np.random.Generator,
) -> list[GroupStanding]:
    """
    Sort standings by: points → overall GD → overall GF → H2H points → H2H GD → lot.
    """
    def sort_key(s: GroupStanding):
        return (s.points, s.gd, s.gf)

    standings.sort(key=sort_key, reverse=True)

    # Resolve ties between teams with identical primary key
    # (simplified: tiebreak by H2H if exactly two teams tied, else random)
    result: list[GroupStanding] = []
    i = 0
    while i < len(standings):
        j = i + 1
        while j < len(standings) and sort_key(standings[j]) == sort_key(standings[i]):
            j += 1
        tied_group = standings[i:j]
        if len(tied_group) == 1:
            result.extend(tied_group)
        else:
            result.extend(_resolve_tie(tied_group, h2h, rng))
        i = j
    return result


def _resolve_tie(
    tied: list[GroupStanding],
    h2h: dict,
    rng: np.random.Generator,
) -> list[GroupStanding]:
    """Apply H2H tiebreaker between two tied teams, then lot."""
    if len(tied) == 2:
        a, b = tied[0].team, tied[1].team
        key = frozenset({a, b})
        if key in h2h:
            pa = h2h[key][a]["pts"]
            pb = h2h[key][b]["pts"]
            if pa > pb:
                return tied
            if pb > pa:
                return [tied[1], tied[0]]
            # H2H GD
            gda = h2h[key][a]["gd"]
            gdb = h2h[key][b]["gd"]
            if gda > gdb:
                return tied
            if gdb > gda:
                return [tied[1], tied[0]]
    # Final: random lot
    shuffled = list(tied)
    rng.shuffle(shuffled)
    return shuffled


def select_best_third_place(
    all_third_place: list[tuple[str, GroupStanding]],
    n: int,
    rng: np.random.Generator,
) -> list[str]:
    """
    From all third-place finishers, select the best `n` to advance.
    Ranking: points → GD → GF → lot.
    """
    def key(item):
        _, s = item
        return (s.points, s.gd, s.gf)

    sorted_third = sorted(all_third_place, key=key, reverse=True)
    advancing: list[str] = []
    i = 0
    while len(advancing) < n and i < len(sorted_third):
        batch_key = key(sorted_third[i])
        batch = [sorted_third[i]]
        while i + 1 < len(sorted_third) and key(sorted_third[i + 1]) == batch_key:
            i += 1
            batch.append(sorted_third[i])

        needed = n - len(advancing)
        if len(batch) <= needed:
            advancing.extend(t for t, _ in batch)
        else:
            # Random draw among tied teams
            selected = [batch[k] for k in rng.choice(len(batch), needed, replace=False)]
            advancing.extend(t for t, _ in selected)
        i += 1
    return advancing
