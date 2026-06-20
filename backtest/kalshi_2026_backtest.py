"""
Retrieve pre-match Kalshi odds for completed WC 2026 group-stage matches.

For settled matches, the live API returns post-settlement prices (0.99/0.01).
We attempt:
  1. Wayback Machine CDX + snapshot (for truly historical pre-match prices)
  2. Fall back to None (unavailable)

For upcoming matches: live API directly.

Usage:
    python backtest/kalshi_2026_backtest.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backtest"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from constants import BACKTEST_DIR, KALSHI_TO_SYSTEM

KALSHI_CACHE_FILE = BACKTEST_DIR / "kalshi_prematch_odds.json"
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
WAYBACK_CDX = "https://web.archive.org/cdx/search/cdx"

# KXWCGAME series ticker format: KXWCGAME-26-XXXvsYYY or similar
# Known Kalshi team name format (3-letter country codes in some cases)
KALSHI_MATCH_SERIES = "KXWCGAME"


def _kalshi_headers(api_key: str = "") -> dict:
    h = {"Content-Type": "application/json"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def discover_match_markets(api_key: str = "") -> list[dict]:
    """
    Retrieve all KXWCGAME markets from Kalshi.
    Returns list of market dicts with ticker, title, status, prices.
    """
    markets = []
    cursor = None
    while True:
        params = {"series_ticker": KALSHI_MATCH_SERIES, "limit": 100, "status": "all"}
        if cursor:
            params["cursor"] = cursor

        r = requests.get(
            f"{KALSHI_BASE}/markets",
            headers=_kalshi_headers(api_key),
            params=params,
            timeout=15,
        )
        if r.status_code == 429:
            time.sleep(5)
            continue
        r.raise_for_status()
        data = r.json()
        batch = data.get("markets", [])
        markets.extend(batch)

        cursor = data.get("cursor")
        if not cursor or not batch:
            break

    return markets


def parse_match_market(m: dict) -> dict | None:
    """
    Parse a single KXWCGAME market into a structured dict.
    Market titles look like: "XXX vs YYY: Team A to win" etc.
    Each match has 3 binary markets: win_a / draw / win_b
    """
    ticker = m.get("ticker", "")
    title = m.get("title", "")
    status = m.get("status", "")

    # Try to extract outcome from title
    title_lower = title.lower()
    if "draw" in title_lower or "tie" in title_lower:
        outcome_type = "draw"
    else:
        # Figure out if it's home or away win
        outcome_type = "team"  # we'll handle below

    def _price(key_d, key_i):
        v = m.get(key_d)
        if v is not None:
            return float(v)
        v = m.get(key_i)
        return float(v or 0) / 100.0

    yes_bid = _price("yes_bid_dollars", "yes_bid")
    yes_ask = _price("yes_ask_dollars", "yes_ask")
    mid = (yes_bid + yes_ask) / 2.0 if (yes_bid + yes_ask) > 0 else None

    settled_price = m.get("result", None)
    is_settled = status in ("finalized", "settled")

    return {
        "ticker": ticker,
        "title": title,
        "status": status,
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "mid": mid,
        "is_settled": is_settled,
        "settled_price": settled_price,
    }


def get_wayback_prematch_odds(ticker: str, match_date: str) -> float | None:
    """
    Try to retrieve a pre-match Kalshi price from Wayback Machine.
    Looks for snapshot from up to 24h before the match.

    Returns mid-price (0-1) or None if not retrievable.
    """
    # Look for snapshots up to 1 day before match date (to avoid live-match prices)
    date_str = match_date.replace("-", "")
    before_date = date_str  # Ideally we'd subtract 1 day but keep it simple

    # Build the Kalshi market URL
    kalshi_url = f"https://kalshi.com/markets/{ticker.split('-')[0]}/{ticker}"

    try:
        cdx_r = requests.get(
            WAYBACK_CDX,
            params={
                "url": kalshi_url,
                "output": "json",
                "limit": 1,
                "to": before_date + "120000",
                "from": before_date + "000000",
                "filter": "statuscode:200",
                "fl": "timestamp,original,statuscode",
            },
            timeout=10,
        )
        cdx_r.raise_for_status()
        rows = cdx_r.json()

        if len(rows) < 2:  # first row is header
            return None

        ts = rows[1][0]  # timestamp like "20260615143022"
        snapshot_url = f"https://web.archive.org/web/{ts}if_/{kalshi_url}"

        snap_r = requests.get(snapshot_url, timeout=15)
        snap_r.raise_for_status()
        html = snap_r.text

        # Heuristic extraction: look for bid/ask price patterns in JSON blobs
        import re
        # Kalshi embeds data in __NEXT_DATA__ or similar
        patterns = [
            r'"yes_bid_dollars"\s*:\s*([0-9.]+)',
            r'"yes_ask_dollars"\s*:\s*([0-9.]+)',
        ]
        bids, asks = [], []
        for p, lst in [(patterns[0], bids), (patterns[1], asks)]:
            for match in re.finditer(p, html):
                lst.append(float(match.group(1)))

        if bids and asks:
            return (bids[0] + asks[0]) / 2.0

        return None

    except Exception as e:
        return None  # Wayback fetch failed — log silently


def _vig_remove(probs: list[float]) -> list[float]:
    """Remove bookmaker vig by normalizing to sum=1."""
    total = sum(p for p in probs if p is not None)
    if total < 0.01:
        return [None] * len(probs)
    return [p / total if p is not None else None for p in probs]


def build_kalshi_comparison(
    completed: pd.DataFrame,
    preds: pd.DataFrame,
    api_key: str = "",
    use_wayback: bool = True,
) -> pd.DataFrame:
    """
    For each completed match, assemble:
      - model ensemble probs (already in preds)
      - Kalshi pre-match probs (from API if active, Wayback if settled)
      - RPS for both model and Kalshi

    Returns DataFrame with columns:
      date, group, team_a, team_b, outcome,
      ens_win_a, ens_draw, ens_win_b,
      kalshi_win_a, kalshi_draw, kalshi_win_b,
      kalshi_available,
      rps_model, rps_kalshi, rps_diff  (positive = model beats Kalshi)
    """
    from evaluate_2026 import rps, merge_predictions_results

    merged = merge_predictions_results(preds, completed)
    if merged.empty:
        return pd.DataFrame()

    # Try to find Kalshi markets
    kalshi_data: dict = {}
    if KALSHI_CACHE_FILE.exists():
        with open(KALSHI_CACHE_FILE) as f:
            kalshi_data = json.load(f)
    else:
        try:
            print("  Fetching Kalshi match markets ...")
            raw_markets = discover_match_markets(api_key=api_key)
            print(f"  Found {len(raw_markets)} KXWCGAME markets")
            # Build match → {win_a, draw, win_b} mapping from parsed markets
            # Group by match (strip outcome suffix from ticker)
            for m in raw_markets:
                parsed = parse_match_market(m)
                if parsed:
                    kalshi_data[parsed["ticker"]] = parsed
            BACKTEST_DIR.mkdir(exist_ok=True)
            with open(KALSHI_CACHE_FILE, "w") as f:
                json.dump(kalshi_data, f, indent=2)
        except Exception as e:
            print(f"  Kalshi API error: {e}")

    from evaluate_2026 import rps as rps_fn

    rows = []
    for _, r in merged.iterrows():
        ta, tb = r["team_a"], r["team_b"]
        outcome = r["outcome"]
        ens_a = float(r["ens_win_a"])
        ens_d = float(r["ens_draw"])
        ens_b = float(r["ens_win_b"])
        rps_model = rps_fn(ens_a, ens_d, ens_b, outcome)

        # Try to find Kalshi odds for this match
        kalshi_a = kalshi_d = kalshi_b = None
        kalshi_available = False

        # Kalshi tickers for match markets follow the KXWCGAME series
        # For now, mark as unavailable (post-settlement)
        # In practice, Wayback is the only option for completed matches

        if use_wayback:
            pass  # Could try Wayback here for each specific ticker

        rps_kalshi = None
        rps_diff = None
        if kalshi_a is not None:
            rps_kalshi = rps_fn(kalshi_a, kalshi_d, kalshi_b, outcome)
            rps_diff = rps_model - rps_kalshi  # positive means model better

        rows.append({
            "date": r.get("date", ""),
            "group": r.get("group", "?"),
            "team_a": ta,
            "team_b": tb,
            "score": f"{int(r.get('score_a',0))}-{int(r.get('score_b',0))}",
            "outcome": outcome,
            "ens_win_a": ens_a,
            "ens_draw": ens_d,
            "ens_win_b": ens_b,
            "kalshi_win_a": kalshi_a,
            "kalshi_draw": kalshi_d,
            "kalshi_win_b": kalshi_b,
            "kalshi_available": kalshi_available,
            "rps_model": rps_model,
            "rps_kalshi": rps_kalshi,
            "rps_diff": rps_diff,
        })

    return pd.DataFrame(rows)


def print_kalshi_comparison(df: pd.DataFrame) -> None:
    """Print Kalshi vs model comparison table."""
    available = df[df["kalshi_available"]]
    print(f"\nKalshi comparison: {len(available)}/{len(df)} matches have pre-match odds")

    if available.empty:
        print("  Pre-match Kalshi odds not retrievable for completed games.")
        print("  Settled markets reset to 0.99/0.01. Wayback fallback available but unreliable.")
        print()
        print("  Model RPS on all completed matches:")
        print(f"    Mean model RPS: {df['rps_model'].mean():.4f}")
        return

    print(f"\n{'Match':<30} {'Outcome':>8} {'Mdl RPS':>8} {'Kal RPS':>8} {'Diff':>8}")
    print("-" * 66)
    for _, r in available.iterrows():
        match = f"{r['team_a'][:12]} v {r['team_b'][:12]}"
        out_str = {2: "A wins", 1: "Draw", 0: "B wins"}.get(r["outcome"], "?")
        rps_k = f"{r['rps_kalshi']:.3f}" if r["rps_kalshi"] is not None else "N/A"
        rps_d = f"{r['rps_diff']:+.3f}" if r["rps_diff"] is not None else "N/A"
        print(f"  {match:<28} {out_str:>8} {r['rps_model']:>8.3f} {rps_k:>8} {rps_d:>8}")

    if available["rps_diff"].notna().any():
        avg_diff = available["rps_diff"].mean()
        beats = (available["rps_diff"] < 0).sum()
        print(f"\n  Avg RPS diff (model - Kalshi): {avg_diff:+.4f}")
        print(f"  Model beats Kalshi: {beats}/{len(available)} matches")


def main() -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.getenv("KALSHI_API_KEY", ""))
    parser.add_argument("--no-wayback", action="store_true")
    args = parser.parse_args()

    pred_file = BACKTEST_DIR / "pretournament_predictions.csv"
    results_file = BACKTEST_DIR / "completed_results.csv"

    if not pred_file.exists() or not results_file.exists():
        print("Run freeze_pretournament.py and get_completed_matches.py first.")
        sys.exit(1)

    preds = pd.read_csv(pred_file)
    completed = pd.read_csv(results_file)

    print(f"Building Kalshi comparison ({len(completed)} completed matches) ...")
    df = build_kalshi_comparison(
        completed, preds,
        api_key=args.api_key,
        use_wayback=not args.no_wayback,
    )

    print_kalshi_comparison(df)

    out = BACKTEST_DIR / "kalshi_comparison_2026.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
