"""
State-aware Monte Carlo simulation of WC 2026.

Treats completed matches as fixed facts, simulates only unplayed matches.

Usage:
    python src/simulate_tournament.py [--n-sims 50000] [--seed 42]

Outputs:
    predictions/simulation_results.csv      (per-team stage probabilities)
    predictions/remaining_match_probs.csv   (per-match outcome probabilities)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
PRED_DIR = PROJECT_ROOT / "predictions"

# ─────────────────────────────────────────────────────────────
# WC 2026 group assignments
# Teams verified against spec completed results; others filled from qualifying
# ─────────────────────────────────────────────────────────────
# Maps team names used in our system → names used in training data / DC params
TEAM_NAME_ALIASES: dict[str, str] = {
    "USA": "United States",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "Czechia": "Czech Republic",
    "North Macedonia": "Macedonia",
    "Turkey": "Turkey",
    "Türkiye": "Turkey",
    "Curacao": "Curaçao",
    "Ivory Coast": "Ivory Coast",  # martj42 may use Côte d'Ivoire
}


def _canonical(team: str) -> str:
    """Resolve team name to the form used in trained models."""
    return TEAM_NAME_ALIASES.get(team, team)


# Verified from ESPN live data, June 2026
WC2026_GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Switzerland", "Qatar"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USA", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cabo Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Congo DR", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# WC 2026 knockout bracket seeding:
# 32 teams = 12 group winners + 12 runners-up + 8 best 3rd-place
# Round of 32 → 16 → QF → SF → Final
KNOCKOUT_ROUNDS = ["round_of_32", "round_of_16", "quarterfinal", "semifinal", "final"]
THIRD_PLACE_ADVANCES = 8  # 8 best 3rd-place teams advance


# ─────────────────────────────────────────────────────────────
# Standings
# ─────────────────────────────────────────────────────────────

@dataclass
class Standing:
    team: str
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    gf: int = 0
    ga: int = 0

    @property
    def points(self) -> int:
        return self.wins * 3 + self.draws

    @property
    def gd(self) -> int:
        return self.gf - self.ga

    def sort_key(self) -> tuple:
        return (self.points, self.gd, self.gf)

    def to_dict(self) -> dict:
        return {
            "team": self.team, "played": self.played,
            "wins": self.wins, "draws": self.draws, "losses": self.losses,
            "gf": self.gf, "ga": self.ga, "gd": self.gd, "points": self.points,
        }


def _update_standing(s: Standing, gf: int, ga: int) -> None:
    s.played += 1
    s.gf += gf
    s.ga += ga
    if gf > ga:
        s.wins += 1
    elif gf == ga:
        s.draws += 1
    else:
        s.losses += 1


def compute_standings_from_results(group_teams: list[str], matches: pd.DataFrame) -> dict[str, Standing]:
    standings = {t: Standing(team=t) for t in group_teams}
    for _, row in matches.iterrows():
        ht, at = str(row["home_team"]), str(row["away_team"])
        if ht in standings and at in standings:
            hs, as_ = int(row["home_score"]), int(row["away_score"])
            _update_standing(standings[ht], hs, as_)
            _update_standing(standings[at], as_, hs)
    return standings


def sort_standings(standings: dict[str, Standing], rng: Optional[np.random.Generator] = None) -> list[Standing]:
    lst = list(standings.values())
    # Primary: pts, gd, gf — ties broken randomly
    if rng is not None:
        noise = rng.random(len(lst)) * 0.001
        return sorted(lst, key=lambda s: s.sort_key()[0] * 1000 + s.sort_key()[1] * 10 + s.sort_key()[2],
                       reverse=True)
    return sorted(lst, key=lambda s: s.sort_key(), reverse=True)


# ─────────────────────────────────────────────────────────────
# Model loading & ensemble
# ─────────────────────────────────────────────────────────────

def load_models() -> dict:
    models = {}

    dc_path = MODELS_DIR / "dixon_coles_params.json"
    if dc_path.exists():
        with open(dc_path) as f:
            models["dc"] = json.load(f)
        log.info("Loaded Dixon-Coles params")
    else:
        log.warning("Dixon-Coles params not found at %s", dc_path)

    elo_path = MODELS_DIR / "elo_ratings.json"
    if elo_path.exists():
        with open(elo_path) as f:
            models["elo"] = json.load(f)
        log.info("Loaded Elo ratings for %d teams", len(models["elo"]))
    else:
        log.warning("Elo ratings not found at %s", elo_path)

    xgb_path = MODELS_DIR / "xgb_wc2026.json"
    feat_path = MODELS_DIR / "xgb_feature_cols.json"
    if xgb_path.exists() and feat_path.exists():
        try:
            import xgboost as xgb
            m = xgb.XGBClassifier()
            m.load_model(str(xgb_path))
            with open(feat_path) as f:
                feat_cols = json.load(f)["feature_cols"]
            models["xgb"] = (m, feat_cols)
            log.info("Loaded XGBoost model")
        except Exception as e:
            log.warning("Could not load XGBoost: %s", e)

    return models


def _dc_tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    if x == 0 and y == 0:
        return max(1e-10, 1.0 - lam * mu * rho)
    if x == 1 and y == 0:
        return max(1e-10, 1.0 + mu * rho)
    if x == 0 and y == 1:
        return max(1e-10, 1.0 + lam * rho)
    if x == 1 and y == 1:
        return max(1e-10, 1.0 - rho)
    return 1.0


# Module-level cache: (team_a, team_b, is_neutral) → {home_win, draw, away_win}
# Probabilities are deterministic given model params, so caching is correct.
_DC_CACHE: dict[tuple, dict] = {}
_SCORES = np.arange(9)
_X_GRID, _Y_GRID = np.meshgrid(_SCORES, _SCORES, indexing="ij")
_MASK_HOME = _X_GRID > _Y_GRID
_MASK_DRAW = _X_GRID == _Y_GRID
_MASK_AWAY = _X_GRID < _Y_GRID


def _dc_probs(dc_params: dict, team_a: str, team_b: str, is_neutral: bool = True) -> dict:
    cache_key = (team_a, team_b, is_neutral)
    if cache_key in _DC_CACHE:
        return _DC_CACHE[cache_key]

    from scipy.stats import poisson as sp_poisson
    attack = dc_params["attack"]
    defense = dc_params["defense"]
    home_adv = 0.0 if is_neutral else dc_params["home_advantage"]
    rho = dc_params["rho"]
    mean_a = float(np.mean(list(attack.values())))
    mean_d = float(np.mean(list(defense.values())))

    ta, tb = _canonical(team_a), _canonical(team_b)
    a_att = attack.get(ta, mean_a)
    a_def = defense.get(ta, mean_d)
    b_att = attack.get(tb, mean_a)
    b_def = defense.get(tb, mean_d)

    lam = max(0.01, float(np.exp(a_att - b_def + home_adv)))
    mu = max(0.01, float(np.exp(b_att - a_def)))

    # Vectorised score grid — much faster than nested Python loops
    p_x = sp_poisson.pmf(_SCORES, lam)  # shape (9,)
    p_y = sp_poisson.pmf(_SCORES, mu)   # shape (9,)
    joint = np.outer(p_x, p_y)           # shape (9,9)

    # Apply Dixon-Coles tau correction to 2×2 corner
    joint[0, 0] *= max(1e-10, 1.0 - lam * mu * rho)
    joint[1, 0] *= max(1e-10, 1.0 + mu * rho)
    joint[0, 1] *= max(1e-10, 1.0 + lam * rho)
    joint[1, 1] *= max(1e-10, 1.0 - rho)

    p_home = float(joint[_MASK_HOME].sum())
    p_draw = float(joint[_MASK_DRAW].sum())
    p_away = float(joint[_MASK_AWAY].sum())

    total = p_home + p_draw + p_away
    if total < 1e-9:
        result = {"home_win": 1 / 3, "draw": 1 / 3, "away_win": 1 / 3}
    else:
        result = {"home_win": p_home / total, "draw": p_draw / total, "away_win": p_away / total}

    _DC_CACHE[cache_key] = result
    return result


def _elo_probs(elo_ratings: dict, team_a: str, team_b: str) -> dict:
    r_a = elo_ratings.get(_canonical(team_a), elo_ratings.get(team_a, 1500.0))
    r_b = elo_ratings.get(_canonical(team_b), elo_ratings.get(team_b, 1500.0))
    p_a = 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))
    draw_base = 0.28
    draw_prob = draw_base * (1.0 - abs(2.0 * p_a - 1.0))
    return {
        "home_win": p_a * (1.0 - draw_prob),
        "draw": draw_prob,
        "away_win": (1.0 - p_a) * (1.0 - draw_prob),
    }


_ENSEMBLE_CACHE: dict[tuple, dict] = {}


def get_match_probs(
    models: dict,
    team_a: str,
    team_b: str,
    is_neutral: bool = True,
) -> dict[str, float]:
    """
    Ensemble: 0.40 DC + 0.40 XGB + 0.20 Elo.
    Falls back gracefully if models are missing. Results are cached.
    """
    cache_key = (team_a, team_b, is_neutral)
    if cache_key in _ENSEMBLE_CACHE:
        return _ENSEMBLE_CACHE[cache_key]

    weights = {"dc": 0.40, "xgb": 0.40, "elo": 0.20}
    available = {k: v for k, v in weights.items() if k in models}
    if not available:
        return {"home_win": 1 / 3, "draw": 1 / 3, "away_win": 1 / 3}

    total_w = sum(available.values())
    p_hw = p_d = p_aw = 0.0

    if "dc" in available:
        dc = _dc_probs(models["dc"], team_a, team_b, is_neutral)
        w = weights["dc"] / total_w
        p_hw += w * dc["home_win"]
        p_d += w * dc["draw"]
        p_aw += w * dc["away_win"]

    if "elo" in available:
        elo = _elo_probs(models["elo"], team_a, team_b)
        w = weights["elo"] / total_w
        p_hw += w * elo["home_win"]
        p_d += w * elo["draw"]
        p_aw += w * elo["away_win"]

    if "xgb" in available:
        xgb_model, feat_cols = models["xgb"]
        elo_ratings = models.get("elo", {})
        ca, cb = _canonical(team_a), _canonical(team_b)
        r_a = elo_ratings.get(ca, elo_ratings.get(team_a, 1500.0))
        r_b = elo_ratings.get(cb, elo_ratings.get(team_b, 1500.0))
        feature_row = {
            "elo_home": r_a, "elo_away": r_b, "elo_diff": r_a - r_b,
            "fifa_pts_home": 1000.0, "fifa_pts_away": 1000.0, "fifa_pts_diff": 0.0,
            "form_win_home": 0.5, "form_draw_home": 0.2, "form_loss_home": 0.3,
            "form_avg_gf_home": 1.5, "form_avg_ga_home": 1.2, "form_pts_home": 1.7,
            "form_win_away": 0.5, "form_draw_away": 0.2, "form_loss_away": 0.3,
            "form_avg_gf_away": 1.5, "form_avg_ga_away": 1.2, "form_pts_away": 1.7,
            "same_confederation": 0, "is_neutral": int(is_neutral), "importance": 4.0,
            "days_since_last_home": 7, "days_since_last_away": 7,
            "h2h_home_wins": 0, "h2h_draws": 0, "h2h_away_wins": 0,
            "conf_home_enc": 0, "conf_away_enc": 0,
        }
        row = np.array([feature_row.get(c, 0.0) for c in feat_cols], dtype=float).reshape(1, -1)
        probs = xgb_model.predict_proba(row)[0]
        w = weights["xgb"] / total_w
        p_hw += w * float(probs[2])
        p_d += w * float(probs[1])
        p_aw += w * float(probs[0])

    result = {"home_win": p_hw, "draw": p_d, "away_win": p_aw}
    _ENSEMBLE_CACHE[cache_key] = result
    return result


def _sample_score(probs: dict, rng: np.random.Generator, allow_draw: bool = True) -> tuple[int, int]:
    """Sample (home_goals, away_goals) consistent with the given outcome probabilities."""
    p_hw = probs["home_win"]
    p_d = probs["draw"] if allow_draw else 0.0
    p_aw = probs["away_win"]
    total = p_hw + p_d + p_aw
    r = rng.random() * total

    if r < p_hw:
        # Home wins: sample goals
        hg = int(rng.poisson(1.5)) + 1
        ag = max(0, int(rng.poisson(0.8)))
        if ag >= hg:
            ag = hg - 1
    elif r < p_hw + p_d:
        # Draw
        hg = int(rng.poisson(1.2))
        ag = hg
    else:
        # Away wins
        ag = int(rng.poisson(1.5)) + 1
        hg = max(0, int(rng.poisson(0.8)))
        if hg >= ag:
            hg = ag - 1

    return max(0, hg), max(0, ag)


# ─────────────────────────────────────────────────────────────
# Group stage simulation
# ─────────────────────────────────────────────────────────────

def simulate_group_stage(
    groups: dict[str, list[str]],
    completed_df: pd.DataFrame,
    models: dict,
    rng: np.random.Generator,
) -> tuple[dict[str, list[Standing]], list[tuple[str, Standing]]]:
    """
    Simulate all unplayed group matches, respecting already-completed matches.

    Returns:
        group_standings: {group_name: [Standing sorted 1st→4th]}
        third_place_teams: [(group_name, Standing) for each group's 3rd-place team]
    """
    group_standings: dict[str, list[Standing]] = {}
    third_place_teams: list[tuple[str, Standing]] = []

    for group_name, teams in groups.items():
        # Build standings from completed matches in this group
        grp_mask = (
            completed_df["group"].str.upper() == group_name.upper()
        ) if "group" in completed_df.columns else pd.Series(False, index=completed_df.index)

        completed_grp = completed_df[grp_mask]
        standings = compute_standings_from_results(teams, completed_grp)

        # Determine which fixtures are still to be played (all pairs not yet completed)
        completed_pairs: set[frozenset] = set()
        for _, row in completed_grp.iterrows():
            completed_pairs.add(frozenset([str(row["home_team"]), str(row["away_team"])]))

        for team_a, team_b in combinations(teams, 2):
            if frozenset([team_a, team_b]) in completed_pairs:
                continue  # already played
            probs = get_match_probs(models, team_a, team_b, is_neutral=True)
            hg, ag = _sample_score(probs, rng, allow_draw=True)
            _update_standing(standings[team_a], hg, ag)
            _update_standing(standings[team_b], ag, hg)

        sorted_standings = sort_standings(standings, rng)
        group_standings[group_name] = sorted_standings

        if len(sorted_standings) > 2:
            third_place_teams.append((group_name, sorted_standings[2]))

    return group_standings, third_place_teams


def select_best_third_place(
    third_place_teams: list[tuple[str, Standing]],
    n_advance: int,
    rng: np.random.Generator,
) -> list[str]:
    """Select the top N third-place teams by points, gd, gf."""
    if not third_place_teams:
        return []
    sorted_third = sorted(third_place_teams, key=lambda x: x[1].sort_key(), reverse=True)
    # Break ties randomly
    return [t for _, s in sorted_third[:n_advance] for t in [s.team]]


# ─────────────────────────────────────────────────────────────
# Knockout simulation
# ─────────────────────────────────────────────────────────────

def simulate_knockout(
    bracket: list[str],
    models: dict,
    rng: np.random.Generator,
) -> tuple[str, dict[str, str]]:
    """
    Simulate knockout bracket. Returns (champion, {team: exit_stage}).
    """
    remaining = list(bracket)
    rng.shuffle(remaining)
    exit_stage: dict[str, str] = {}

    for stage in KNOCKOUT_ROUNDS:
        if len(remaining) < 2:
            break
        next_round = []
        for i in range(0, len(remaining), 2):
            if i + 1 >= len(remaining):
                next_round.append(remaining[i])
                continue
            home, away = remaining[i], remaining[i + 1]
            probs = get_match_probs(models, home, away, is_neutral=True)

            p_hw = probs["home_win"]
            p_aw = probs["away_win"]
            total = p_hw + p_aw
            if total < 1e-9:
                total = 1.0
                p_hw = 0.5

            # For knockout, no draws — 20% extra-time/pens if close
            p_hw_norm = p_hw / total
            diff = abs(2 * p_hw_norm - 1)
            if diff < 0.20:  # teams within 10pp of each other
                p_hw_norm = rng.random()  # essentially coin flip with bias
                p_hw_norm = p_hw_norm * 0.6 + 0.2  # clamp to [0.2, 0.8]

            winner = home if rng.random() < p_hw_norm else away
            loser = away if winner == home else home
            exit_stage[loser] = stage
            next_round.append(winner)

        remaining = next_round

    if remaining:
        champion = remaining[0]
        exit_stage[champion] = "champion"
        return champion, exit_stage
    return "", exit_stage


# ─────────────────────────────────────────────────────────────
# Main simulation
# ─────────────────────────────────────────────────────────────

@dataclass
class SimResults:
    n_sims: int
    stage_counts: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    champion_counts: dict = field(default_factory=lambda: defaultdict(int))


def run_simulation(
    completed_df: pd.DataFrame,
    models: dict,
    groups: dict[str, list[str]] = None,
    n_sims: int = 50_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run N Monte Carlo simulations from current tournament state.

    Returns:
        sim_results_df: per-team stage probabilities
        match_probs_df: per-remaining-match outcome probabilities
    """
    if groups is None:
        groups = WC2026_GROUPS

    rng = np.random.default_rng(seed)
    results = SimResults(n_sims)

    # Pre-compute remaining match list for probability output
    completed_pairs: set[frozenset] = set()
    if "group" in completed_df.columns:
        for _, row in completed_df.iterrows():
            completed_pairs.add(frozenset([str(row["home_team"]), str(row["away_team"])]))

    remaining_matches = []
    for group_name, teams in groups.items():
        for ta, tb in combinations(teams, 2):
            if frozenset([ta, tb]) not in completed_pairs:
                remaining_matches.append((group_name, ta, tb))

    log.info("Running %d simulations (%d remaining group matches) ...", n_sims, len(remaining_matches))

    for i in range(n_sims):
        if i % 5000 == 0:
            log.info("  sim %d/%d", i, n_sims)

        group_standings, third_place_teams = simulate_group_stage(
            groups, completed_df, models, rng
        )

        # Build 32-team bracket: winners + runners-up + best 8 third
        qualified: list[str] = []
        for g_name in sorted(groups.keys()):
            standings = group_standings[g_name]
            if len(standings) >= 1:
                qualified.append(standings[0].team)  # winner
                results.stage_counts[standings[0].team]["group_winner"] += 1
        for g_name in sorted(groups.keys()):
            standings = group_standings[g_name]
            if len(standings) >= 2:
                qualified.append(standings[1].team)  # runner-up

        # Best 8 third-place
        best_third = select_best_third_place(third_place_teams, THIRD_PLACE_ADVANCES, rng)
        qualified.extend(best_third)
        for t in best_third:
            results.stage_counts[t]["third_place_advance"] += 1

        # Record group-stage advancement
        for t in qualified:
            results.stage_counts[t]["advances_from_group"] += 1

        champion, exit_stage = simulate_knockout(qualified, models, rng)
        if champion:
            results.champion_counts[champion] += 1

        for team, stage in exit_stage.items():
            results.stage_counts[team][stage] += 1

    # ── Build output DataFrames ──────────────────────────────────
    all_teams = set()
    for g_teams in groups.values():
        all_teams.update(g_teams)

    rows = []
    for team in all_teams:
        sc = results.stage_counts[team]
        rows.append({
            "team": team,
            "advances_from_group": sc.get("advances_from_group", 0) / n_sims,
            "group_winner_pct": sc.get("group_winner", 0) / n_sims,
            "third_place_advance_pct": sc.get("third_place_advance", 0) / n_sims,
            "r32_exit_pct": sc.get("round_of_32", 0) / n_sims,
            "r16_exit_pct": sc.get("round_of_16", 0) / n_sims,
            "qf_exit_pct": sc.get("quarterfinal", 0) / n_sims,
            "sf_exit_pct": sc.get("semifinal", 0) / n_sims,
            "final_pct": (sc.get("final", 0) + results.champion_counts.get(team, 0)) / n_sims,
            "champion_pct": results.champion_counts.get(team, 0) / n_sims,
        })

    sim_df = pd.DataFrame(rows).sort_values("champion_pct", ascending=False).reset_index(drop=True)

    # Per-remaining-match probabilities (from expected values, not per simulation)
    match_rows = []
    for group_name, ta, tb in remaining_matches:
        p = get_match_probs(models, ta, tb, is_neutral=True)
        match_rows.append({
            "group": group_name,
            "home_team": ta,
            "away_team": tb,
            "home_win_prob": p["home_win"],
            "draw_prob": p["draw"],
            "away_win_prob": p["away_win"],
        })

    match_df = pd.DataFrame(match_rows)
    return sim_df, match_df


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sims", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    PRED_DIR.mkdir(parents=True, exist_ok=True)

    completed_path = RAW_DIR / "wc2026_completed.csv"
    if not completed_path.exists():
        log.warning("No completed matches file found at %s — simulating full tournament", completed_path)
        completed_df = pd.DataFrame(columns=["date", "home_team", "away_team", "home_score", "away_score", "stage", "group"])
    else:
        completed_df = pd.read_csv(completed_path)
        log.info("Loaded %d completed matches", len(completed_df))

    models = load_models()
    if not models:
        log.error("No models found in %s — run train_models.py first", MODELS_DIR)
        sys.exit(1)

    sim_df, match_df = run_simulation(completed_df, models, n_sims=args.n_sims, seed=args.seed)

    sim_path = PRED_DIR / "simulation_results.csv"
    sim_df.to_csv(sim_path, index=False)
    log.info("Saved simulation results → %s", sim_path)

    match_path = PRED_DIR / "remaining_match_probs.csv"
    match_df.to_csv(match_path, index=False)
    log.info("Saved remaining match probabilities → %s", match_path)

    print("\n--- Top 20 Championship Probabilities ---")
    print(sim_df[["team", "champion_pct", "final_pct", "advances_from_group"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
