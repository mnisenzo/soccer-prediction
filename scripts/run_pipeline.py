"""
End-to-end pipeline: fit models, run simulation, evaluate markets.

Usage:
    python scripts/run_pipeline.py [--model elo_logistic|poisson_goals] [--n-sims 10000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from soccer_prediction.data.database import get_session, init_db
from soccer_prediction.data.repositories import get_completed_matches, get_fixtures, get_group_assignments
from soccer_prediction.markets.evaluator import MarketEvaluator
from soccer_prediction.markets.registry import MarketRegistry
from soccer_prediction.models.registry import ModelRegistry, list_models
from soccer_prediction.simulation.monte_carlo import MonteCarloSimulator
from soccer_prediction.simulation.tournament import DEFAULT_WC2026_GROUPS, TournamentConfig

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
MARKETS_DIR = CONFIGS_DIR / "markets"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="elo_logistic", choices=list_models())
    parser.add_argument("--n-sims", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    init_db()

    # 1. Load data
    with get_session() as session:
        matches = get_completed_matches(session)
        group_assignments = get_group_assignments(session, "FIFA World Cup", 2026)

    if matches.empty:
        print("No completed matches found. Run scripts/load_data.py first.")
        sys.exit(1)

    print(f"Loaded {len(matches)} historical matches.")

    # 2. Fit model
    registry = ModelRegistry()
    model = registry.fit_and_register(args.model, matches)
    print(f"Fitted model: {model.name}")

    # 3. Build tournament config
    if not group_assignments:
        print("No group assignments in DB — using default WC 2026 groups.")
        group_assignments = DEFAULT_WC2026_GROUPS

    config = TournamentConfig.world_cup_2026(group_assignments)

    # 4. Run Monte Carlo simulation
    print(f"Running {args.n_sims} simulations...")
    sim = MonteCarloSimulator(n_simulations=args.n_sims, seed=args.seed)

    def progress(i, n):
        if i % max(1, n // 10) == 0:
            print(f"  {i}/{n} ({100*i//n}%)")

    results = sim.run(config, model, progress_callback=progress)

    # 5. Print team probabilities
    probs_df = results.team_probabilities()
    print("\n--- Tournament Winner Probabilities ---")
    top = probs_df[["team", "wins_tournament", "advances_from_group"]].head(16)
    for _, row in top.iterrows():
        print(f"  {row['team']:<20} win={row['wins_tournament']:.1%}   adv={row['advances_from_group']:.1%}")

    # 6. Load and evaluate markets
    mkt_registry = MarketRegistry()
    n_markets = mkt_registry.load_directory(MARKETS_DIR)
    print(f"\nLoaded {n_markets} markets from {MARKETS_DIR}")

    evaluator = MarketEvaluator(sim_results=results, model=model)
    market_results = evaluator.evaluate_all(mkt_registry.all())

    print("\n--- Market Probabilities ---")
    print(f"{'Market':<45} {'Model':>7} {'Market':>8} {'Edge':>7}")
    print("-" * 70)
    for mr in sorted(market_results, key=lambda x: x.model_probability, reverse=True):
        edge_str = f"{mr.edge:+.1%}" if mr.edge is not None else "   -"
        mkt_str = f"{mr.market_price:.1%}" if mr.market_price else "   -"
        print(f"  {mr.market.name:<43} {mr.model_probability:.1%}  {mkt_str:>7}  {edge_str:>7}")

    # 7. Export
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    probs_df.to_csv(out_dir / "team_probabilities.csv", index=False)

    mkt_df = pd.DataFrame([mr.to_dict() for mr in market_results])
    mkt_df.to_csv(out_dir / "market_probabilities.csv", index=False)
    print(f"\nExported results to {out_dir}/")


if __name__ == "__main__":
    main()
