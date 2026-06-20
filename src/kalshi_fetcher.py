"""
Fetch live WC 2026 prediction market odds from Kalshi.

No authentication required for read-only market data.

Usage:
    python src/kalshi_fetcher.py
    python src/kalshi_fetcher.py --save   # save to predictions/kalshi_odds.json
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
PRED_DIR = PROJECT_ROOT / "predictions"
BASE_URL = "https://external-api.kalshi.com/trade-api/v2"

_SESSION = requests.Session()
_SESSION.headers.update({"Accept": "application/json"})


def _get(endpoint: str, params: dict = None) -> Optional[dict]:
    url = f"{BASE_URL}{endpoint}"
    try:
        r = _SESSION.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        log.warning("HTTP %s for %s: %s", e.response.status_code, url, e)
        return None
    except requests.RequestException as e:
        log.warning("Request error for %s: %s", url, e)
        return None


def discover_wc_tickers() -> list[dict]:
    """
    Find all World Cup series tickers by browsing the /series endpoint.
    Filters for soccer/football world cup events.
    """
    data = _get("/series", {"category": "sports", "limit": 200})
    if not data:
        return []

    series_list = data.get("series") or data.get("data") or []
    wc_series = []
    for s in series_list:
        title = (s.get("title") or "").lower()
        if "world cup" in title or "wc" in title or "soccer" in title:
            wc_series.append({
                "ticker": s.get("ticker"),
                "title": s.get("title"),
                "category": s.get("category"),
            })

    log.info("Found %d WC-related series: %s", len(wc_series), [s["ticker"] for s in wc_series])
    return wc_series


def _parse_markets(markets: list[dict]) -> list[dict]:
    """Parse Kalshi market list into normalised rows.

    Kalshi v2 API uses *_dollars fields pre-scaled to [0,1] (not 0-100).
    Falls back to *_bid/*_ask if dollars fields are absent.
    """
    rows = []
    for m in markets:
        # Try dollars fields first (v2 API); fall back to integer fields / 100
        def _price(key_dollars: str, key_int: str) -> float:
            v = m.get(key_dollars)
            if v is not None:
                return float(v)
            v = m.get(key_int)
            return float(v or 0) / 100.0

        yes_bid = _price("yes_bid_dollars", "yes_bid")
        yes_ask = _price("yes_ask_dollars", "yes_ask")
        last = _price("last_price_dollars", "last_price")

        # Use last price as mid when bid=0 (thinly traded markets)
        if yes_bid == 0 and yes_ask > 0:
            mid = (yes_ask + last) / 2 if last > 0 else yes_ask
        else:
            mid = (yes_bid + yes_ask) / 2

        volume_fp = m.get("volume_fp") or m.get("volume_24h_fp") or m.get("volume") or 0

        rows.append({
            "ticker": m.get("ticker", ""),
            "title": m.get("title", ""),
            "yes_sub_title": m.get("yes_sub_title", ""),
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "last_price": last,
            "kalshi_mid": mid,
            "volume": float(volume_fp),
            "status": m.get("status", ""),
            "close_time": m.get("close_time", ""),
        })
    return rows


def get_wc_winner_odds() -> pd.DataFrame:
    """
    Fetch World Cup winner contracts.
    Primary series: KXMENWORLDCUP
    """
    series_to_try = ["KXMENWORLDCUP", "KXWORLDCUP2026", "KXWC26WIN"]

    for series in series_to_try:
        data = _get("/markets", {"series_ticker": series, "status": "open", "limit": 100})
        if data and data.get("markets"):
            markets = data["markets"]
            log.info("Found %d winner markets under series %s", len(markets), series)
            rows = _parse_markets(markets)
            df = pd.DataFrame(rows)
            if df.empty:
                continue

            # Extract team name from yes_sub_title or title
            df["team"] = df["yes_sub_title"].where(df["yes_sub_title"].str.len() > 0, df["title"])

            # Remove vig: normalise probabilities
            total_mid = df["kalshi_mid"].sum()
            if total_mid > 0:
                df["kalshi_prob"] = df["kalshi_mid"] / total_mid
            else:
                df["kalshi_prob"] = df["kalshi_mid"]

            return df[["team", "ticker", "yes_bid", "yes_ask", "kalshi_mid", "kalshi_prob",
                        "volume", "close_time"]]

    log.warning("Could not find WC winner markets. Checked: %s", series_to_try)
    return pd.DataFrame()


_KALSHI_TEAM_NORM: dict[str, str] = {
    "usa": "USA", "united states": "USA", "us": "USA",
    "south korea": "South Korea", "korea republic": "South Korea", "korea": "South Korea",
    "côte d'ivoire": "Ivory Coast", "cote d'ivoire": "Ivory Coast", "ivory coast": "Ivory Coast",
    "bosnia and herzegovina": "Bosnia and Herzegovina", "bosnia": "Bosnia and Herzegovina",
    "bosnia-herzegovina": "Bosnia and Herzegovina",
    "ir iran": "Iran", "iran": "Iran",
    "new zealand": "New Zealand",
    "cabo verde": "Cabo Verde", "cape verde": "Cabo Verde",
    "saudi arabia": "Saudi Arabia",
    "dr congo": "Congo DR", "congo dr": "Congo DR", "congo": "Congo DR",
    "czechia": "Czechia", "czech republic": "Czechia",
    "curacao": "Curacao", "curaçao": "Curacao",
    "mexico": "Mexico", "germany": "Germany", "france": "France",
    "spain": "Spain", "argentina": "Argentina", "brazil": "Brazil",
    "england": "England", "portugal": "Portugal", "netherlands": "Netherlands",
    "canada": "Canada", "japan": "Japan", "australia": "Australia",
    "switzerland": "Switzerland", "belgium": "Belgium", "croatia": "Croatia",
    "italy": "Italy", "uruguay": "Uruguay", "sweden": "Sweden",
    "norway": "Norway", "morocco": "Morocco", "senegal": "Senegal",
    "ghana": "Ghana", "ecuador": "Ecuador", "paraguay": "Paraguay",
    "austria": "Austria", "colombia": "Colombia", "bolivia": "Bolivia",
    "tunisia": "Tunisia", "algeria": "Algeria", "egypt": "Egypt",
    "iraq": "Iraq", "jordan": "Jordan", "qatar": "Qatar",
    "panama": "Panama", "jamaica": "Jamaica", "mali": "Mali",
    "south africa": "South Africa",
    "turkey": "Turkey", "turkiye": "Turkey", "türkiye": "Turkey",
    "scotland": "Scotland", "haiti": "Haiti", "uzbekistan": "Uzbekistan",
}


def _norm_team(name: str) -> str:
    return _KALSHI_TEAM_NORM.get(name.lower().strip(), name.strip())


def parse_match_markets(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Group raw KXWCGAME market data into per-match 3-way (win/draw/win) odds.

    Kalshi format (observed June 2026):
      - title:         "Congo DR vs Uzbekistan Winner?"
      - yes_sub_title: "Congo DR" | "Uzbekistan" | "Tie"
      - event_ticker:  "KXWCGAME-26JUN27CODUZB"  (3 markets share this)

    Returns DataFrame with:
        team_a, team_b, kalshi_win_a, kalshi_draw, kalshi_win_b, vig
    """
    import re

    if raw_df.empty:
        return pd.DataFrame()

    df = raw_df.copy()

    # Group by event_ticker (all 3 markets for a match share one event ticker)
    if "event_ticker" in df.columns and df["event_ticker"].notna().any():
        df["_event"] = df["event_ticker"].fillna("")
    else:
        df["_event"] = df["ticker"].str.rsplit("-", n=1).str[0]

    # Strip " Winner?" suffix and similar noise from title
    TITLE_NOISE = re.compile(r'\s*Winner\??\s*$', re.I)
    VS_RE = re.compile(r'^(.+?)\s+vs\.?\s+(.+?)$', re.I)
    # Tie/Draw detection in yes_sub_title
    DRAW_RE = re.compile(r'^(tie|draw)$', re.I)

    rows = []
    skipped = []
    for event_key, group in df.groupby("_event"):
        if len(group) < 3:
            skipped.append((event_key, f"only {len(group)} markets"))
            continue

        # ── Find draw market (yes_sub_title == "Tie" or "Draw") ──────────────
        draw_mask = group["yes_sub_title"].apply(
            lambda s: bool(DRAW_RE.match(str(s or "")))
        )
        draw_rows = group[draw_mask]
        win_rows = group[~draw_mask]

        if draw_rows.empty or len(win_rows) < 2:
            skipped.append((event_key, f"draw={len(draw_rows)} win={len(win_rows)}"))
            continue

        draw_price = float(draw_rows.iloc[0]["kalshi_mid"])

        # ── Extract team names from title ─────────────────────────────────────
        raw_title = str(group["title"].iloc[0])
        clean_title = TITLE_NOISE.sub("", raw_title).strip()
        m = VS_RE.match(clean_title)
        if not m:
            skipped.append((event_key, f"no vs in title: {raw_title!r}"))
            continue

        team_a = _norm_team(m.group(1).strip())
        team_b = _norm_team(m.group(2).strip())

        # ── Match win markets to teams via yes_sub_title ──────────────────────
        # yes_sub_title is exactly the team name (e.g. "Congo DR", "USA", "Turkiye")
        win_a = win_b = None
        for _, mkt in win_rows.iterrows():
            sub_raw = str(mkt.get("yes_sub_title", "") or "")
            sub_norm = _norm_team(sub_raw)
            price = float(mkt["kalshi_mid"])

            if sub_norm == team_a or sub_raw == team_a:
                win_a = price
            elif sub_norm == team_b or sub_raw == team_b:
                win_b = price
            else:
                # Fallback: fuzzy first-word match
                sub_lower = sub_norm.lower()
                ta_first = team_a.lower().split()[0]
                tb_first = team_b.lower().split()[0]
                if ta_first in sub_lower and ta_first not in tb_first:
                    win_a = price
                elif tb_first in sub_lower and tb_first not in ta_first:
                    win_b = price
                elif win_a is None:
                    win_a = price
                else:
                    win_b = price

        if win_a is None or win_b is None:
            skipped.append((event_key, f"could not assign wins: subs={[r['yes_sub_title'] for _,r in win_rows.iterrows()]} teams=({team_a},{team_b})"))
            continue

        total = win_a + draw_price + win_b
        if total < 0.01:
            continue

        rows.append({
            "team_a": team_a,
            "team_b": team_b,
            "kalshi_win_a": round(win_a / total, 4),
            "kalshi_draw":  round(draw_price / total, 4),
            "kalshi_win_b": round(win_b / total, 4),
            "kalshi_win_a_raw": win_a,
            "kalshi_draw_raw":  draw_price,
            "kalshi_win_b_raw": win_b,
            "vig": round(total - 1.0, 4),
            "n_markets": len(group),
            "event_ticker": event_key,
        })

    if skipped:
        log.debug("Skipped %d events: %s", len(skipped), skipped[:5])

    result = pd.DataFrame(rows)
    log.info("parse_match_markets: %d events → %d matched", len(df.groupby("_event")), len(result))
    return result


def get_wc_match_odds(parse: bool = True) -> pd.DataFrame:
    """
    Fetch head-to-head match markets from KXWCGAME series.
    If parse=True, returns per-match 3-way odds. Otherwise raw market rows.
    """
    series_to_try = ["KXWCGAME", "KXWCMATCH", "KXWCH2H26", "KXWC26GAME", "KXWC2026"]

    for series in series_to_try:
        data = _get("/markets", {"series_ticker": series, "status": "open", "limit": 200})
        if data and data.get("markets"):
            markets = data["markets"]
            log.info("Found %d match markets under series %s", len(markets), series)
            rows = _parse_markets(markets)
            raw_df = pd.DataFrame(rows)
            if raw_df.empty:
                continue
            if parse:
                parsed = parse_match_markets(raw_df)
                if not parsed.empty:
                    log.info("Parsed %d matches from %d markets", len(parsed), len(raw_df))
                    return parsed
                log.warning("Parsing returned 0 matches — returning raw data")
                return raw_df
            return raw_df

    log.warning("Could not find WC match markets. Checked: %s", series_to_try)
    return pd.DataFrame()


def get_wc_group_odds() -> pd.DataFrame:
    """
    Fetch group winner / qualifier markets.
    """
    series_to_try = ["KXWCGROUP", "KXWCGROUPWIN", "KXWC26GRP"]

    for series in series_to_try:
        data = _get("/markets", {"series_ticker": series, "status": "open", "limit": 200})
        if data and data.get("markets"):
            markets = data["markets"]
            log.info("Found %d group markets under series %s", len(markets), series)
            rows = _parse_markets(markets)
            df = pd.DataFrame(rows)
            if not df.empty:
                return df

    log.warning("Could not find WC group markets. Checked: %s", series_to_try)
    return pd.DataFrame()


def get_all_kalshi_wc_data() -> dict:
    """
    Fetch all WC 2026 Kalshi market data.
    Returns dict with 'winner', 'matches', 'groups', 'tickers', 'fetched_at'.
    """
    log.info("Discovering WC tickers ...")
    tickers = discover_wc_tickers()

    log.info("Fetching winner odds ...")
    winner_df = get_wc_winner_odds()

    log.info("Fetching match odds ...")
    match_df = get_wc_match_odds()

    log.info("Fetching group odds ...")
    group_df = get_wc_group_odds()

    return {
        "winner": winner_df,
        "matches": match_df,
        "groups": group_df,
        "tickers": tickers,
        "fetched_at": datetime.utcnow().isoformat(),
    }


def to_american_odds(p: float) -> str:
    """Convert implied probability to American odds string."""
    if p <= 0 or p >= 1:
        return "N/A"
    if p > 0.5:
        odds = -(p / (1 - p)) * 100
        return f"{odds:.0f}"
    else:
        odds = ((1 - p) / p) * 100
        return f"+{odds:.0f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save results to predictions/")
    args = parser.parse_args()

    data = get_all_kalshi_wc_data()

    print(f"\nFetched at: {data['fetched_at']}")
    print(f"Tickers found: {[t['ticker'] for t in data['tickers']]}")

    winner_df = data["winner"]
    if not winner_df.empty:
        print(f"\n── WC Winner Markets ({len(winner_df)} contracts) ──")
        winner_df["american_odds"] = winner_df["kalshi_prob"].apply(to_american_odds)
        print(winner_df[["team", "kalshi_prob", "american_odds", "volume"]].head(20).to_string(index=False))
    else:
        print("\nNo winner markets found.")

    match_df = data["matches"]
    if not match_df.empty:
        print(f"\n── Match Markets ({len(match_df)} contracts) ──")
        print(match_df[["title", "kalshi_mid", "volume"]].head(20).to_string(index=False))
    else:
        print("\nNo match markets found.")

    if args.save:
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        out = {
            "tickers": data["tickers"],
            "fetched_at": data["fetched_at"],
            "winner": winner_df.to_dict(orient="records") if not winner_df.empty else [],
            "matches": match_df.to_dict(orient="records") if not match_df.empty else [],
            "groups": data["groups"].to_dict(orient="records") if not data["groups"].empty else [],
        }
        path = PRED_DIR / "kalshi_odds.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        log.info("Saved Kalshi data → %s", path)


if __name__ == "__main__":
    main()
