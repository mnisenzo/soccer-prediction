"""
Load teams, historical matches, and WC 2026 fixtures into the database.

Usage:
    python scripts/load_data.py [--reset]

Options:
    --reset   Drop and recreate all tables before loading (fresh start).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from soccer_prediction.data.database import init_db, get_engine
from soccer_prediction.data.loaders import DataLoader
from soccer_prediction.data.schema import Base

SAMPLE_DIR = Path(__file__).parent.parent / "data" / "sample"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database before loading")
    args = parser.parse_args()

    if args.reset:
        print("Resetting database...")
        engine = get_engine()
        Base.metadata.drop_all(engine)

    init_db()
    loader = DataLoader()

    teams_csv = SAMPLE_DIR / "teams.csv"
    if teams_csv.exists():
        n = loader.load_teams(teams_csv)
        print(f"Teams loaded: {n} new")
    else:
        print(f"WARNING: {teams_csv} not found. Run generate_sample_data.py first.")
        sys.exit(1)

    historical_csv = SAMPLE_DIR / "historical_matches.csv"
    if historical_csv.exists():
        n = loader.load_historical_matches(historical_csv)
        print(f"Historical matches loaded: {n}")
    else:
        print(f"WARNING: {historical_csv} not found. Run: python scripts/generate_sample_data.py")

    fixtures_csv = SAMPLE_DIR / "wc_2026_fixtures.csv"
    if fixtures_csv.exists():
        n = loader.load_fixtures(fixtures_csv, "FIFA World Cup", 2026)
        print(f"WC 2026 fixtures loaded: {n}")
    else:
        print(f"WARNING: {fixtures_csv} not found.")

    print("Done.")


if __name__ == "__main__":
    main()
