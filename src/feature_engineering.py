"""
Feature engineering pipeline for WC 2026 prediction model.

Inputs:
  data/raw/international_results.csv   (from martj42/international_results)
  data/raw/rankings/*.csv              (from Kaggle cashncarry/fifaworldranking)

Outputs:
  data/processed/features.parquet

Usage:
    python src/feature_engineering.py
"""
from __future__ import annotations

import logging
import sys
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TOURNAMENT_K = {
    "FIFA World Cup": 50,
    "FIFA World Cup qualification": 30,
    "UEFA Euro": 40,
    "Copa América": 40,
    "Africa Cup of Nations": 40,
    "Asian Cup": 40,
    "CONCACAF Gold Cup": 40,
    "Friendly": 15,
}

TOURNAMENT_IMPORTANCE = {
    "FIFA World Cup": 4.0,
    "FIFA World Cup qualification": 1.5,
    "UEFA Euro": 2.0,
    "Copa América": 2.0,
    "Africa Cup of Nations": 2.0,
    "Asian Cup": 2.0,
    "CONCACAF Gold Cup": 2.0,
    "Friendly": 0.5,
}

CONFEDERATION_MAP = {
    "Germany": "UEFA", "France": "UEFA", "Spain": "UEFA", "Portugal": "UEFA",
    "Netherlands": "UEFA", "Belgium": "UEFA", "England": "UEFA", "Italy": "UEFA",
    "Croatia": "UEFA", "Austria": "UEFA", "Denmark": "UEFA", "Switzerland": "UEFA",
    "Sweden": "UEFA", "Norway": "UEFA", "Poland": "UEFA", "Czechia": "UEFA",
    "Serbia": "UEFA", "Scotland": "UEFA", "Turkey": "UEFA", "Romania": "UEFA",
    "Bosnia and Herzegovina": "UEFA", "Slovakia": "UEFA", "Hungary": "UEFA",
    "Ukraine": "UEFA", "Finland": "UEFA", "Albania": "UEFA",
    "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "Colombia": "CONMEBOL",
    "Uruguay": "CONMEBOL", "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL",
    "Bolivia": "CONMEBOL", "Venezuela": "CONMEBOL", "Peru": "CONMEBOL",
    "Chile": "CONMEBOL", "Algeria": "CONMEBOL",
    "USA": "CONCACAF", "Canada": "CONCACAF", "Mexico": "CONCACAF",
    "Panama": "CONCACAF", "Jamaica": "CONCACAF", "Costa Rica": "CONCACAF",
    "Curacao": "CONCACAF", "Haiti": "CONCACAF",
    "Japan": "AFC", "South Korea": "AFC", "Iran": "AFC", "Saudi Arabia": "AFC",
    "Australia": "AFC", "Qatar": "AFC", "Iraq": "AFC",
    "Senegal": "CAF", "Morocco": "CAF", "Egypt": "CAF", "Nigeria": "CAF",
    "Ivory Coast": "CAF", "South Africa": "CAF", "Ghana": "CAF",
    "Tunisia": "CAF", "Congo DR": "CAF", "Mali": "CAF", "Cameroon": "CAF",
    "Algeria": "CAF", "Cabo Verde": "CAF",
    "New Zealand": "OFC",
    "Jordan": "AFC", "Czechia": "UEFA",
}


def _k_factor(tournament: str) -> float:
    for key, k in TOURNAMENT_K.items():
        if key.lower() in tournament.lower():
            return k
    return 20.0


def _importance(tournament: str) -> float:
    for key, v in TOURNAMENT_IMPORTANCE.items():
        if key.lower() in tournament.lower():
            return v
    return 1.0


def compute_elo_ratings(results_df: pd.DataFrame) -> dict[str, float]:
    """
    Compute current Elo ratings by replaying match history from 1990 onward.

    Returns dict mapping team name → current Elo rating.
    """
    ratings: dict[str, float] = {}
    cutoff = pd.Timestamp("1990-01-01")

    df = results_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= cutoff].sort_values("date")

    for _, row in df.iterrows():
        home = str(row["home_team"])
        away = str(row["away_team"])
        h_score = row.get("home_score", None)
        a_score = row.get("away_score", None)

        if pd.isna(h_score) or pd.isna(a_score):
            continue

        h_score, a_score = int(h_score), int(a_score)
        r_h = ratings.get(home, 1500.0)
        r_a = ratings.get(away, 1500.0)

        is_neutral = bool(row.get("neutral", False))
        home_adv = 0.0 if is_neutral else 75.0
        k = _k_factor(str(row.get("tournament", "")))

        # WC final/SF boosted
        stage = str(row.get("stage", "")).lower()
        if "final" in stage or "semi" in stage:
            if "world cup" in str(row.get("tournament", "")).lower():
                k = 60

        exp_h = 1.0 / (1.0 + 10.0 ** (-((r_h + home_adv) - r_a) / 400.0))
        if h_score > a_score:
            s_h = 1.0
        elif h_score == a_score:
            s_h = 0.5
        else:
            s_h = 0.0

        ratings[home] = r_h + k * (s_h - exp_h)
        ratings[away] = r_a + k * ((1.0 - s_h) - (1.0 - exp_h))

    return ratings


def compute_rolling_form(
    results_df: pd.DataFrame,
    team: str,
    as_of_date: pd.Timestamp,
    n_matches: int = 10,
    comp_only: bool = False,
) -> dict:
    """
    Compute rolling form stats for a team before a given date.
    """
    df = results_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    mask = (
        ((df["home_team"] == team) | (df["away_team"] == team))
        & (df["date"] < as_of_date)
    )
    if comp_only:
        mask &= ~df["tournament"].str.lower().str.contains("friendly", na=False)

    recent = df[mask].sort_values("date").tail(n_matches)

    if recent.empty:
        return {
            "win_rate": 0.0, "draw_rate": 0.0, "loss_rate": 0.0,
            "avg_gf": 0.0, "avg_ga": 0.0, "avg_gd": 0.0,
            "pts_per_game": 0.0, "n_matches": 0,
        }

    wins = draws = losses = 0
    gf_list, ga_list = [], []

    for _, row in recent.iterrows():
        h_score = row.get("home_score", None)
        a_score = row.get("away_score", None)
        if pd.isna(h_score) or pd.isna(a_score):
            continue
        h_score, a_score = int(h_score), int(a_score)

        if row["home_team"] == team:
            gf, ga = h_score, a_score
        else:
            gf, ga = a_score, h_score

        gf_list.append(gf)
        ga_list.append(ga)
        if gf > ga:
            wins += 1
        elif gf == ga:
            draws += 1
        else:
            losses += 1

    n = wins + draws + losses
    if n == 0:
        return {
            "win_rate": 0.0, "draw_rate": 0.0, "loss_rate": 0.0,
            "avg_gf": 0.0, "avg_ga": 0.0, "avg_gd": 0.0,
            "pts_per_game": 0.0, "n_matches": 0,
        }

    avg_gf = float(np.mean(gf_list)) if gf_list else 0.0
    avg_ga = float(np.mean(ga_list)) if ga_list else 0.0

    return {
        "win_rate": wins / n,
        "draw_rate": draws / n,
        "loss_rate": losses / n,
        "avg_gf": avg_gf,
        "avg_ga": avg_ga,
        "avg_gd": avg_gf - avg_ga,
        "pts_per_game": (3 * wins + draws) / n,
        "n_matches": n,
    }


def get_fifa_points(rankings_df: pd.DataFrame, team: str, as_of_date: pd.Timestamp) -> float:
    """
    Return FIFA ranking points for team as of most recent ranking date <= as_of_date.
    """
    if rankings_df is None or rankings_df.empty:
        return 1000.0

    team_rows = rankings_df[rankings_df["country_full"] == team].copy()
    if team_rows.empty:
        # Try partial name match
        team_rows = rankings_df[
            rankings_df["country_full"].str.lower() == team.lower()
        ].copy()
    if team_rows.empty:
        return 1000.0

    team_rows["rank_date"] = pd.to_datetime(team_rows["rank_date"])
    past = team_rows[team_rows["rank_date"] <= as_of_date]
    if past.empty:
        past = team_rows  # fallback: use earliest available

    return float(past.sort_values("rank_date").iloc[-1]["total_points"])


def _last_match_date(results_df: pd.DataFrame, team: str, before: pd.Timestamp) -> Optional[pd.Timestamp]:
    mask = (
        ((results_df["home_team"] == team) | (results_df["away_team"] == team))
        & (results_df["date"] < before)
    )
    rows = results_df[mask]
    if rows.empty:
        return None
    return rows["date"].max()


def _h2h_last5(results_df: pd.DataFrame, team_a: str, team_b: str, before: pd.Timestamp) -> tuple:
    mask = (
        (
            ((results_df["home_team"] == team_a) & (results_df["away_team"] == team_b))
            | ((results_df["home_team"] == team_b) & (results_df["away_team"] == team_a))
        )
        & (results_df["date"] < before)
    )
    h2h = results_df[mask].sort_values("date").tail(5)

    a_wins = draws = b_wins = 0
    for _, row in h2h.iterrows():
        hs, as_ = row.get("home_score"), row.get("away_score")
        if pd.isna(hs) or pd.isna(as_):
            continue
        hs, as_ = int(hs), int(as_)
        if row["home_team"] == team_a:
            gf, ga = hs, as_
        else:
            gf, ga = as_, hs
        if gf > ga:
            a_wins += 1
        elif gf == ga:
            draws += 1
        else:
            b_wins += 1

    return a_wins, draws, b_wins


def build_feature_matrix(
    results_df: pd.DataFrame,
    rankings_df: Optional[pd.DataFrame] = None,
    start_year: int = 2000,
    min_importance: float = 1.0,
) -> pd.DataFrame:
    """
    Build feature matrix for all matches from start_year onward with importance >= min_importance.
    """
    log.info("Computing Elo ratings from full history ...")
    # We'll compute Elo incrementally as we iterate; build a snapshot dict
    # For efficiency: pre-compute progressive Elo at each match
    df = results_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Progressive Elo
    elo_snapshot: dict[str, float] = {}
    elo_at_match: list[tuple[float, float]] = []  # (elo_home, elo_away) before update

    for _, row in df.iterrows():
        home = str(row["home_team"])
        away = str(row["away_team"])
        r_h = elo_snapshot.get(home, 1500.0)
        r_a = elo_snapshot.get(away, 1500.0)
        elo_at_match.append((r_h, r_a))

        h_score = row.get("home_score")
        a_score = row.get("away_score")
        if pd.isna(h_score) or pd.isna(a_score):
            continue
        h_score, a_score = int(h_score), int(a_score)

        is_neutral = bool(row.get("neutral", False))
        home_adv = 0.0 if is_neutral else 75.0
        k = _k_factor(str(row.get("tournament", "")))

        exp_h = 1.0 / (1.0 + 10.0 ** (-((r_h + home_adv) - r_a) / 400.0))
        s_h = 1.0 if h_score > a_score else (0.5 if h_score == a_score else 0.0)

        elo_snapshot[home] = r_h + k * (s_h - exp_h)
        elo_snapshot[away] = r_a + k * ((1.0 - s_h) - (1.0 - exp_h))

    df["elo_home_before"] = [e[0] for e in elo_at_match]
    df["elo_away_before"] = [e[1] for e in elo_at_match]

    # Filter to training window
    cutoff = pd.Timestamp(f"{start_year}-01-01")
    df["importance"] = df["tournament"].apply(lambda t: _importance(str(t)))
    train_df = df[
        (df["date"] >= cutoff)
        & (df["importance"] >= min_importance)
        & df["home_score"].notna()
        & df["away_score"].notna()
    ].copy()

    log.info("Building features for %d training matches ...", len(train_df))

    rows = []
    for idx, row in train_df.iterrows():
        home = str(row["home_team"])
        away = str(row["away_team"])
        match_date = row["date"]

        # Form features
        h_form = compute_rolling_form(df, home, match_date, n_matches=10)
        a_form = compute_rolling_form(df, away, match_date, n_matches=10)

        # FIFA points
        h_pts = get_fifa_points(rankings_df, home, match_date) if rankings_df is not None else 1000.0
        a_pts = get_fifa_points(rankings_df, away, match_date) if rankings_df is not None else 1000.0

        # Confederations
        h_conf = CONFEDERATION_MAP.get(home, "OTHER")
        a_conf = CONFEDERATION_MAP.get(away, "OTHER")

        # Days since last match
        h_last = _last_match_date(df, home, match_date)
        a_last = _last_match_date(df, away, match_date)
        h_days = (match_date - h_last).days if h_last else 90
        a_days = (match_date - a_last).days if a_last else 90

        # H2H
        h2h_a, h2h_d, h2h_b = _h2h_last5(df, home, away, match_date)

        # Target
        h_score, a_score = int(row["home_score"]), int(row["away_score"])
        if h_score > a_score:
            target = 2  # home win
        elif h_score == a_score:
            target = 1  # draw
        else:
            target = 0  # away win

        rows.append({
            "date": match_date,
            "home_team": home,
            "away_team": away,
            "elo_home": row["elo_home_before"],
            "elo_away": row["elo_away_before"],
            "elo_diff": row["elo_home_before"] - row["elo_away_before"],
            "fifa_pts_home": h_pts,
            "fifa_pts_away": a_pts,
            "fifa_pts_diff": h_pts - a_pts,
            "form_win_home": h_form["win_rate"],
            "form_draw_home": h_form["draw_rate"],
            "form_loss_home": h_form["loss_rate"],
            "form_avg_gf_home": h_form["avg_gf"],
            "form_avg_ga_home": h_form["avg_ga"],
            "form_pts_home": h_form["pts_per_game"],
            "form_win_away": a_form["win_rate"],
            "form_draw_away": a_form["draw_rate"],
            "form_loss_away": a_form["loss_rate"],
            "form_avg_gf_away": a_form["avg_gf"],
            "form_avg_ga_away": a_form["avg_ga"],
            "form_pts_away": a_form["pts_per_game"],
            "confederation_home": h_conf,
            "confederation_away": a_conf,
            "same_confederation": int(h_conf == a_conf),
            "is_neutral": int(bool(row.get("neutral", False))),
            "importance": row["importance"],
            "days_since_last_home": h_days,
            "days_since_last_away": a_days,
            "h2h_home_wins": h2h_a,
            "h2h_draws": h2h_d,
            "h2h_away_wins": h2h_b,
            "home_score": h_score,
            "away_score": a_score,
            "tournament": str(row.get("tournament", "")),
            "target": target,
        })

    features_df = pd.DataFrame(rows)
    log.info("Feature matrix: %d rows, %d cols", len(features_df), len(features_df.columns))
    return features_df


def main() -> None:
    results_path = RAW_DIR / "international_results.csv"
    if not results_path.exists():
        log.error("Missing %s — run: curl -o %s https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
                  results_path, results_path)
        sys.exit(1)

    log.info("Loading international results ...")
    results_df = pd.read_csv(results_path)
    results_df.columns = results_df.columns.str.strip()
    # Normalise column names (martj42 dataset uses 'home_score'/'away_score')
    if "home_score" not in results_df.columns and "home_goals" in results_df.columns:
        results_df = results_df.rename(columns={"home_goals": "home_score", "away_goals": "away_score"})

    # Optional FIFA rankings
    rankings_df = None
    ranking_files = list((RAW_DIR / "rankings").glob("*.csv"))
    if ranking_files:
        log.info("Loading FIFA rankings from %d files ...", len(ranking_files))
        rankings_df = pd.concat([pd.read_csv(f) for f in ranking_files], ignore_index=True)
        rankings_df.columns = rankings_df.columns.str.strip()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features_df = build_feature_matrix(results_df, rankings_df)
    out = PROCESSED_DIR / "features.parquet"
    features_df.to_parquet(out, index=False)
    log.info("Saved features to %s", out)


if __name__ == "__main__":
    main()
