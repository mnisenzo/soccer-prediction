"""
Compare model championship probabilities against pre-tournament betting odds.

Usage:
    python backtest/odds_comparison.py

Output:
    backtest/odds_comparison.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "backtest"))

PRED_DIR     = PROJECT_ROOT / "predictions"
BACKTEST_DIR = PROJECT_ROOT / "backtest"
OUT_FILE     = BACKTEST_DIR / "odds_comparison.csv"

# Pre-tournament American odds (positive = underdog).
# All teams are positive odds (underdogs vs a hypothetical favourite).
BETTING_ODDS_AMERICAN: dict[str, int] = {
    "France":               380,
    "Spain":                550,
    "England":              550,
    "Argentina":            850,
    "Portugal":            1000,
    "Brazil":              1200,
    "Germany":             1300,
    "Netherlands":         1800,
    "Norway":              3000,
    "Morocco":             3000,
    "USA":                 3500,
    "Belgium":             4000,
    "Colombia":            5000,
    "Japan":               5000,
    "Mexico":              5500,
    "Switzerland":         6500,
    "Croatia":             9000,
    "Senegal":            10000,
    "Uruguay":            10000,
    "Sweden":             12000,
    "Ecuador":            12000,
    "South Korea":        15000,
    "Canada":             15000,
    "Austria":            16000,
    "Australia":          20000,
    "Ivory Coast":        20000,
    "Scotland":           30000,
    "Algeria":            30000,
    "Czech Republic":     35000,
    "Egypt":              40000,
    "Paraguay":           45000,
    "Ghana":              50000,
    "Bosnia-Herzegovina": 70000,
    "DR Congo":           80000,
    "Saudi Arabia":       80000,
    "Iran":              100000,
    "Tunisia":           150000,
    "South Africa":      200000,
    "Panama":            200000,
    "Uzbekistan":        250000,
    "Cape Verde":        250000,
    "Qatar":             250000,
    "Curacao":           250000,
    "New Zealand":       250000,
    "Jordan":            250000,
    "Iraq":              250000,
}

# Reconcile simulation names → odds names where they differ
SIM_TO_ODDS: dict[str, str] = {
    "United States": "USA",
    "Cabo Verde":    "Cape Verde",
    "Congo DR":      "DR Congo",
    "Czechia":       "Czech Republic",
    "Curaçao":       "Curacao",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
}


def american_to_implied_prob(odds: int) -> float:
    """Positive American odds → raw implied probability."""
    return 100.0 / (odds + 100.0)


def remove_vig(raw_probs: dict[str, float]) -> tuple[dict[str, float], float]:
    total = sum(raw_probs.values())
    return {t: p / total for t, p in raw_probs.items()}, total


def build_comparison_table() -> pd.DataFrame:
    sim_path = PRED_DIR / "simulation_results.csv"
    if not sim_path.exists():
        print(f"ERROR: {sim_path} not found. Run src/simulate_tournament.py first.")
        sys.exit(1)

    sim = pd.read_csv(sim_path)
    # Normalise sim team names to odds names
    sim["odds_name"] = sim["team"].map(lambda t: SIM_TO_ODDS.get(t, t))
    sim_dict = dict(zip(sim["odds_name"], sim["champion_pct"]))

    # Compute raw and fair implied probabilities
    raw_probs = {t: american_to_implied_prob(o) for t, o in BETTING_ODDS_AMERICAN.items()}
    fair_probs, overround = remove_vig(raw_probs)
    print(f"Overround: {overround:.4f}  ({(overround - 1) * 100:.1f}% vig)")

    rows = []
    unmatched_sim   = []
    unmatched_odds  = []

    for team, odds in BETTING_ODDS_AMERICAN.items():
        model_prob = sim_dict.get(team)
        if model_prob is None:
            unmatched_odds.append(team)
            model_prob = float("nan")

        fair = fair_probs[team]
        edge = (model_prob - fair) if model_prob == model_prob else float("nan")

        rows.append({
            "team":             team,
            "american_odds":    f"+{odds}",
            "raw_implied_prob": round(raw_probs[team], 5),
            "fair_implied_prob": round(fair, 5),
            "model_prob":       round(float(model_prob), 5) if model_prob == model_prob else float("nan"),
            "edge":             round(edge, 5) if edge == edge else float("nan"),
        })

    # Flag sim teams not in odds list
    for t in sim["odds_name"]:
        if t not in BETTING_ODDS_AMERICAN:
            unmatched_sim.append(t)

    if unmatched_odds:
        print(f"WARNING: {len(unmatched_odds)} betting teams not in simulation: {unmatched_odds}")
    if unmatched_sim:
        print(f"INFO: {len(unmatched_sim)} simulation teams not in odds (ignored): {unmatched_sim}")

    df = pd.DataFrame(rows).sort_values("model_prob", ascending=False).reset_index(drop=True)
    return df, overround


def print_comparison_table(df: pd.DataFrame, overround: float) -> None:
    print()
    print("=" * 60)
    print("=== CHAMPIONSHIP ODDS COMPARISON (all 48 teams) ===")
    print(f"Overround: {overround:.4f}  ({(overround - 1) * 100:.1f}% vig)")
    print()
    print(f"  {'Team':<24} {'Model%':>7}  {'Market%':>8}  {'Edge':>7}  {'Odds':>9}")
    print("  " + "-" * 58)

    for _, r in df.iterrows():
        mp   = r["model_prob"]
        mkt  = r["fair_implied_prob"]
        edge = r["edge"]
        mp_s   = f"{mp:.1%}"   if mp == mp   else "—"
        mkt_s  = f"{mkt:.1%}"
        edge_s = f"{edge:+.1%}" if edge == edge else "—"
        print(f"  {r['team']:<24} {mp_s:>7}  {mkt_s:>8}  {edge_s:>7}  {r['american_odds']:>9}")

    print()
    valid = df.dropna(subset=["edge"])

    top_pos = valid.nlargest(3, "edge")
    print("Largest positive edges (model > market):")
    for i, (_, r) in enumerate(top_pos.iterrows(), 1):
        print(f"  {i}. {r['team']:<22} {r['edge']:+.2%}")

    top_neg = valid.nsmallest(3, "edge")
    print()
    print("Largest negative edges (market > model):")
    for i, (_, r) in enumerate(top_neg.iterrows(), 1):
        print(f"  {i}. {r['team']:<22} {r['edge']:+.2%}")

    flagged = valid[valid["edge"].abs() > 0.03]
    if not flagged.empty:
        print()
        print(f"Teams with |edge| > 3% — review for model artifacts:")
        for _, r in flagged.iterrows():
            print(f"  {r['team']:<22} edge={r['edge']:+.2%}  model={r['model_prob']:.1%}  market={r['fair_implied_prob']:.1%}")


def main() -> None:
    df, overround = build_comparison_table()
    print_comparison_table(df, overround)

    df.to_csv(OUT_FILE, index=False)
    print(f"\nSaved -> {OUT_FILE}")


if __name__ == "__main__":
    main()
