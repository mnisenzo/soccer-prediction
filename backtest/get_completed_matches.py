"""
Load completed 2026 WC group-stage results and normalize team names.

Primary source: data/raw/wc2026_completed.csv (already maintained by the pipeline).
Optional fallback: football-data.org API.

Usage:
    python backtest/get_completed_matches.py [--api-key KEY]
    python backtest/get_completed_matches.py --update   # fetch from API and update CSV

Output:
    backtest/completed_results.csv  (canonical names, outcome column)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backtest"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from constants import BACKTEST_DIR, RAW_DIR, to_system_name

OUT_FILE = BACKTEST_DIR / "completed_results.csv"


def outcome_code(home_score: int, away_score: int) -> int:
    """Return 2=home(team_a) win, 1=draw, 0=away(team_b) win."""
    if home_score > away_score:
        return 2
    if home_score == away_score:
        return 1
    return 0


def load_from_csv() -> pd.DataFrame:
    """Load and normalize from data/raw/wc2026_completed.csv."""
    path = RAW_DIR / "wc2026_completed.csv"
    if not path.exists():
        raise FileNotFoundError(f"No completed matches file at {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # Rename columns to canonical form
    df = df.rename(columns={
        "home_team": "team_a",
        "away_team": "team_b",
        "home_score": "score_a",
        "away_score": "score_b",
    })

    # Add outcome and goal columns
    df["outcome"]     = df.apply(lambda r: outcome_code(int(r["score_a"]), int(r["score_b"])), axis=1)
    df["total_goals"] = df["score_a"].astype(int) + df["score_b"].astype(int)
    df["goal_diff"]   = df["score_a"].astype(int) - df["score_b"].astype(int)

    # Normalize team names (handle any discrepancies)
    df["team_a"] = df["team_a"].apply(lambda n: to_system_name(n, "fd"))
    df["team_b"] = df["team_b"].apply(lambda n: to_system_name(n, "fd"))

    # Build match_id (sort teams alphabetically for consistent key)
    def make_id(row):
        grp = str(row.get("group", "?")).strip()
        ta, tb = row["team_a"], row["team_b"]
        return f"{grp}_{ta.replace(' ','_')}_{tb.replace(' ','_')}"

    df["match_id"] = df.apply(make_id, axis=1)
    df["source"] = "wc2026_completed.csv"

    cols = ["match_id", "date", "group", "team_a", "team_b",
            "score_a", "score_b", "outcome", "total_goals", "goal_diff",
            "stage", "source"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].sort_values("date").reset_index(drop=True)


def fetch_from_api(api_key: str) -> pd.DataFrame:
    """Fetch from football-data.org and normalize team names."""
    import requests

    headers = {"X-Auth-Token": api_key}
    r = requests.get(
        "https://api.football-data.org/v4/competitions/WC/matches",
        params={"season": 2026, "status": "FINISHED"},
        headers=headers,
        timeout=15,
    )
    r.raise_for_status()
    raw_matches = r.json().get("matches", [])
    print(f"  API returned {len(raw_matches)} finished matches")

    rows = []
    for m in raw_matches:
        stage_raw = m.get("stage", "GROUP_STAGE")
        if "GROUP" not in stage_raw:
            continue  # only group stage for now

        score_ft = m.get("score", {}).get("fullTime", {})
        sa = score_ft.get("home")
        sb = score_ft.get("away")
        if sa is None or sb is None:
            continue

        group_raw = m.get("group", "") or ""
        group = group_raw.replace("GROUP_", "").strip() or "?"

        ta = to_system_name(m["homeTeam"]["name"], "fd")
        tb = to_system_name(m["awayTeam"]["name"], "fd")
        date = m["utcDate"][:10]
        mid = f"{group}_{ta.replace(' ','_')}_{tb.replace(' ','_')}"

        rows.append({
            "match_id": mid,
            "date": date,
            "group": group,
            "team_a": ta,
            "team_b": tb,
            "score_a": int(sa),
            "score_b": int(sb),
            "outcome": outcome_code(int(sa), int(sb)),
            "stage": "Group Stage",
            "source": "football-data.org",
        })

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def get_completed_2026_matches(
    update_from_api: bool = False,
    api_key: str = "",
) -> pd.DataFrame:
    """
    Load completed 2026 WC results in canonical form.

    If update_from_api=True, fetches from football-data.org and updates
    the raw CSV (requires api_key).
    """
    if update_from_api and api_key:
        print("Fetching from football-data.org ...")
        api_df = fetch_from_api(api_key)
        if not api_df.empty:
            # Merge with existing CSV — API is authoritative for finished matches
            raw_path = RAW_DIR / "wc2026_completed.csv"
            # Write to raw CSV for pipeline use
            api_out = api_df.rename(columns={
                "team_a": "home_team", "team_b": "away_team",
                "score_a": "home_score", "score_b": "away_score",
            })[["date", "home_team", "away_team", "home_score", "away_score", "stage", "group"]]
            api_out.to_csv(raw_path, index=False)
            print(f"  Updated {raw_path} with {len(api_out)} matches")

    df = load_from_csv()
    OUT_FILE.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_FILE, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.getenv("FOOTBALL_DATA_API_KEY", ""))
    parser.add_argument("--update", action="store_true", help="Fetch from API and update")
    args = parser.parse_args()

    df = get_completed_2026_matches(
        update_from_api=args.update,
        api_key=args.api_key,
    )

    print(f"\nCompleted 2026 WC matches: {len(df)}")
    print(df[["date", "group", "team_a", "team_b", "score_a", "score_b", "outcome"]].to_string(index=False))
    print(f"\nOutcome distribution: {df['outcome'].value_counts().to_dict()}")
    print(f"  2=A win, 1=draw, 0=B win")


if __name__ == "__main__":
    main()
