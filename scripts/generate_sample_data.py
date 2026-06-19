"""
Generate synthetic historical match data for bootstrapping the system.

Usage:
    python scripts/generate_sample_data.py [--n-matches 300] [--seed 42]

Produces: data/sample/historical_matches.csv
"""
from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from soccer_prediction.features.elo import DEFAULT_ELO

TEAMS_CSV = Path(__file__).parent.parent / "data" / "sample" / "teams.csv"
OUTPUT_CSV = Path(__file__).parent.parent / "data" / "sample" / "historical_matches.csv"

TOURNAMENTS = [
    ("FIFA World Cup", 2022),
    ("UEFA Euro", 2024),
    ("Copa America", 2024),
    ("CONMEBOL Qualifiers", 2023),
    ("UEFA Nations League", 2023),
    ("CAF CAN", 2024),
    ("AFC Asian Cup", 2023),
    ("CONCACAF Nations League", 2023),
    ("International Friendly", 2022),
    ("International Friendly", 2023),
    ("International Friendly", 2024),
]


def goals_from_lambda(lam: float, rng: np.random.Generator) -> int:
    return int(rng.poisson(lam))


def simulate_match(
    home_elo: float,
    away_elo: float,
    avg_goals: float,
    home_advantage: float,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Sample goals using Poisson model derived from Elo ratings."""
    elo_ratio = 10 ** ((home_elo + home_advantage - away_elo) / 400)
    attack_h = elo_ratio / (1 + elo_ratio) * 2  # scale to ~1
    attack_a = 1 / (1 + elo_ratio) * 2

    lam_h = attack_h * avg_goals
    lam_a = attack_a * avg_goals

    return goals_from_lambda(lam_h, rng), goals_from_lambda(lam_a, rng)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-matches", type=int, default=350)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    teams_df = pd.read_csv(TEAMS_CSV)
    teams = teams_df.set_index("name")["elo_rating"].to_dict()
    team_names = list(teams.keys())

    start_date = datetime(2021, 6, 1)
    end_date = datetime(2025, 6, 1)
    date_range = (end_date - start_date).days

    records = []
    for _ in range(args.n_matches):
        home, away = rng.choice(team_names, size=2, replace=False)
        t_name, t_year = TOURNAMENTS[int(rng.integers(len(TOURNAMENTS)))]
        days_offset = int(rng.integers(date_range))
        match_date = start_date + timedelta(days=days_offset)
        is_neutral = bool(rng.random() > 0.3)

        home_elo = teams.get(home, DEFAULT_ELO)
        away_elo = teams.get(away, DEFAULT_ELO)
        home_adv = 0.0 if is_neutral else 100.0

        hg, ag = simulate_match(home_elo, away_elo, 1.25, home_adv, rng)

        records.append({
            "home_team": home,
            "away_team": away,
            "match_date": match_date.strftime("%Y-%m-%d"),
            "home_goals": hg,
            "away_goals": ag,
            "tournament": t_name,
            "is_neutral": is_neutral,
            "stage": "group" if "World Cup" in t_name or "Euro" in t_name else "friendly",
        })

    df = pd.DataFrame(records).sort_values("match_date")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Generated {len(df)} matches -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
