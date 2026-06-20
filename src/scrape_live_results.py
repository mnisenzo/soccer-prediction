"""
Scrape live WC 2026 match results from football-data.org API.

Usage:
    python src/scrape_live_results.py [--api-key YOUR_KEY]

Free tier: https://www.football-data.org/client/register
Set API key via env var FOOTBALL_DATA_API_KEY or --api-key flag.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent
OUT_FILE = PROJECT_ROOT / "data" / "raw" / "wc2026_completed.csv"
API_BASE = "https://api.football-data.org/v4"


def fetch_wc_matches(api_key: str) -> list[dict]:
    url = f"{API_BASE}/competitions/WC/matches"
    params = {"season": 2026}
    headers = {"X-Auth-Token": api_key}
    r = requests.get(url, params=params, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json().get("matches", [])


def parse_matches(raw: list[dict]) -> pd.DataFrame:
    rows = []
    for m in raw:
        status = m.get("status", "")
        if status not in ("FINISHED",):
            continue

        score = m.get("score", {})
        ft = score.get("fullTime", {})
        home_score = ft.get("home")
        away_score = ft.get("away")
        if home_score is None or away_score is None:
            continue

        group = m.get("group", "") or ""
        # football-data returns "GROUP_A" style
        if group.startswith("GROUP_"):
            group = group.replace("GROUP_", "")
        elif not group:
            group = "?"

        stage_raw = m.get("stage", "GROUP_STAGE")
        stage = "Group Stage" if "GROUP" in stage_raw else stage_raw.replace("_", " ").title()

        rows.append({
            "date": m["utcDate"][:10],
            "home_team": m["homeTeam"]["name"],
            "away_team": m["awayTeam"]["name"],
            "home_score": int(home_score),
            "away_score": int(away_score),
            "stage": stage,
            "group": group,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("date").reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape WC 2026 live results")
    parser.add_argument("--api-key", default=os.getenv("FOOTBALL_DATA_API_KEY", ""))
    parser.add_argument("--out", default=str(OUT_FILE))
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: No API key provided.")
        print("  Set env var FOOTBALL_DATA_API_KEY=<key>  or pass --api-key <key>")
        print("  Get a free key at https://www.football-data.org/client/register")
        sys.exit(1)

    print("Fetching WC 2026 matches from football-data.org ...")
    try:
        raw = fetch_wc_matches(args.api_key)
    except requests.HTTPError as e:
        print(f"HTTP error: {e}")
        sys.exit(1)

    df = parse_matches(raw)
    if df.empty:
        print("No finished matches found yet.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} completed matches to {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
