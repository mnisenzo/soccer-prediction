"""
End-to-end WC 2026 prediction pipeline.

Steps:
  1. scrape_live_results.py   — fetch latest completed match results
  2. feature_engineering.py  — build feature matrix from historical data
  3. train_models.py          — fit Dixon-Coles, XGBoost, Elo models
  4. simulate_tournament.py   — run 50k Monte Carlo simulations
  5. (optional) streamlit_app.py — launch dashboard

Usage:
    python run_pipeline.py [--skip-scrape] [--skip-features] [--skip-xgb]
                           [--n-sims 50000] [--api-key YOUR_KEY] [--dashboard]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"


def run(script: str, extra_args: list[str] = None, desc: str = "") -> bool:
    cmd = [sys.executable, str(SRC_DIR / script)] + (extra_args or [])
    print(f"\n{'='*60}")
    print(f"  {desc or script}")
    print(f"  {' '.join(cmd)}")
    print("=" * 60)
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"\nERROR: {script} failed (exit {result.returncode})")
        return False
    return True


def download_results_csv() -> bool:
    """Download international_results.csv if not present."""
    out = PROJECT_ROOT / "data" / "raw" / "international_results.csv"
    if out.exists():
        print(f"  Already exists: {out}")
        return True
    print("  Downloading international_results.csv ...")
    try:
        import urllib.request
        url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
        urllib.request.urlretrieve(url, out)
        print(f"  Saved to {out}")
        return True
    except Exception as e:
        print(f"  ERROR downloading: {e}")
        print(f"  Manual download: curl -o {out} {url}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="WC 2026 prediction pipeline")
    parser.add_argument("--skip-scrape", action="store_true",
                        help="Skip live results scraping")
    parser.add_argument("--skip-features", action="store_true",
                        help="Skip feature engineering (use existing features.parquet)")
    parser.add_argument("--skip-xgb", action="store_true",
                        help="Skip XGBoost training (much faster without Optuna tuning)")
    parser.add_argument("--n-sims", type=int, default=50_000)
    parser.add_argument("--api-key", default=os.getenv("FOOTBALL_DATA_API_KEY", ""),
                        help="football-data.org API key for live results")
    parser.add_argument("--xgb-trials", type=int, default=50)
    parser.add_argument("--dashboard", action="store_true",
                        help="Launch Streamlit dashboard after pipeline completes")
    args = parser.parse_args()

    print("WC 2026 Prediction Pipeline")
    print("=" * 60)

    # ── Step 0: Download training data ────────────────────────────
    print("\nStep 0: Checking training data ...")
    (PROJECT_ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
    if not download_results_csv():
        print("WARNING: Could not download results.csv — feature engineering may fail")

    # ── Step 1: Live results ───────────────────────────────────────
    if not args.skip_scrape:
        if args.api_key:
            ok = run("scrape_live_results.py",
                     ["--api-key", args.api_key],
                     desc="Step 1: Scrape live WC 2026 results")
        else:
            print("\nStep 1: SKIPPED (no --api-key; using existing wc2026_completed.csv)")
    else:
        print("\nStep 1: SKIPPED (--skip-scrape)")

    # ── Step 2: Feature engineering ───────────────────────────────
    if not args.skip_features:
        ok = run("feature_engineering.py", desc="Step 2: Build feature matrix")
        if not ok:
            print("Feature engineering failed — Dixon-Coles will still run; XGBoost will be skipped")
    else:
        print("\nStep 2: SKIPPED (--skip-features)")

    # ── Step 3: Train models ───────────────────────────────────────
    train_args = []
    if args.skip_xgb:
        train_args.append("--skip-xgb")
    else:
        train_args += ["--xgb-trials", str(args.xgb_trials)]

    ok = run("train_models.py", train_args, desc="Step 3: Train models (DC + XGBoost + Elo)")
    if not ok:
        print("Model training failed — cannot continue")
        sys.exit(1)

    # ── Step 4: Simulate tournament ────────────────────────────────
    ok = run("simulate_tournament.py",
             ["--n-sims", str(args.n_sims), "--seed", "42"],
             desc=f"Step 4: Run {args.n_sims:,} Monte Carlo simulations")
    if not ok:
        print("Simulation failed")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"  Simulation results: {PROJECT_ROOT / 'predictions' / 'simulation_results.csv'}")
    print(f"  Match probs:        {PROJECT_ROOT / 'predictions' / 'remaining_match_probs.csv'}")
    print("")
    print("Next steps:")
    print("  streamlit run streamlit_app.py")
    print("  python src/kalshi_fetcher.py --save")

    # ── Step 5: Launch dashboard ───────────────────────────────────
    if args.dashboard:
        print("\nLaunching Streamlit dashboard ...")
        subprocess.run(
            ["streamlit", "run", str(PROJECT_ROOT / "streamlit_app.py")],
            cwd=str(PROJECT_ROOT),
        )


if __name__ == "__main__":
    main()
