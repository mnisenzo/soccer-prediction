"""
Shared constants for the 2026 WC backtest suite.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
BACKTEST_DIR = PROJECT_ROOT / "backtest"
MODELS_DIR = PROJECT_ROOT / "models"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# ─── Team name reconciliation ─────────────────────────────────────────────────
# Maps external-source names → canonical names used in this system
# football-data.org / Kalshi → our WC2026_GROUPS names

FOOTBALL_DATA_TO_SYSTEM: dict[str, str] = {
    # football-data.org / ESPN display names → our system names
    "United States": "USA",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia Herzegovina": "Bosnia and Herzegovina",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Cabo Verde": "Cabo Verde",
    "Cape Verde": "Cabo Verde",       # ESPN uses "Cape Verde"
    "DR Congo": "Congo DR",
    "Congo DR": "Congo DR",
    "Czech Republic": "Czechia",
    "Curaçao": "Curacao",
    "Türkiye": "Turkey",
    "Türkiye": "Turkey",          # Unicode for Türkiye
    "Ivory Coast": "Ivory Coast",
    "Scotland": "Scotland",
    "Haiti": "Haiti",
    "Uzbekistan": "Uzbekistan",
    "Australia": "Australia",
    "Turkey": "Turkey",
}

KALSHI_TO_SYSTEM: dict[str, str] = {
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Czech Republic": "Czechia",
    "Curaçao": "Curacao",
    "DR Congo": "Congo DR",
    "United States": "USA",
    "Ivory Coast": "Ivory Coast",
}

# Maps our system names → names in the trained models (martj42 dataset)
SYSTEM_TO_TRAINING: dict[str, str] = {
    "USA": "United States",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "Czechia": "Czech Republic",
    "Curacao": "Curaçao",
    "Turkey": "Turkey",           # martj42 uses "Turkey" not "Türkiye"
    "Ivory Coast": "Ivory Coast", # martj42 may use "Ivory Coast" or "Côte d'Ivoire"
}


def to_system_name(name: str, source: str = "fd") -> str:
    """Convert external name to canonical system name."""
    name = str(name).strip()
    mapping = FOOTBALL_DATA_TO_SYSTEM if source == "fd" else KALSHI_TO_SYSTEM
    return mapping.get(name, name)


def to_training_name(name: str) -> str:
    """Convert system name to name used in trained models."""
    return SYSTEM_TO_TRAINING.get(str(name), name)


# ─── WC 2026 groups (verified from ESPN live data, June 2026) ─────────────────
WC2026_GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Switzerland", "Qatar"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USA", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cabo Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Congo DR", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Known completed match dates (from ESPN, verified June 2026)
KNOWN_MATCH_DATES: dict[tuple, str] = {
    ("A", "Mexico", "South Africa"): "2026-06-11",
    ("A", "South Korea", "Czechia"): "2026-06-12",
    ("A", "Czechia", "South Africa"): "2026-06-18",
    ("A", "Mexico", "South Korea"): "2026-06-19",
    ("B", "Canada", "Bosnia and Herzegovina"): "2026-06-12",
    ("B", "Qatar", "Switzerland"): "2026-06-13",
    ("B", "Switzerland", "Bosnia and Herzegovina"): "2026-06-18",
    ("B", "Canada", "Qatar"): "2026-06-18",
    ("C", "Brazil", "Morocco"): "2026-06-13",
    ("C", "Haiti", "Scotland"): "2026-06-14",
    ("C", "Scotland", "Morocco"): "2026-06-19",
    ("C", "Brazil", "Haiti"): "2026-06-20",
    ("D", "USA", "Paraguay"): "2026-06-13",
    ("D", "Australia", "Turkey"): "2026-06-14",
    ("D", "USA", "Australia"): "2026-06-19",
    ("D", "Turkey", "Paraguay"): "2026-06-20",
    ("E", "Germany", "Curacao"): "2026-06-14",
    ("E", "Ivory Coast", "Ecuador"): "2026-06-14",
    ("F", "Netherlands", "Japan"): "2026-06-14",
    ("F", "Sweden", "Tunisia"): "2026-06-15",
    ("G", "Belgium", "Egypt"): "2026-06-15",
    ("G", "Iran", "New Zealand"): "2026-06-16",
    ("H", "Spain", "Cabo Verde"): "2026-06-15",
    ("H", "Saudi Arabia", "Uruguay"): "2026-06-15",
    ("I", "France", "Senegal"): "2026-06-16",
    ("I", "Iraq", "Norway"): "2026-06-16",
    ("J", "Argentina", "Algeria"): "2026-06-17",
    ("J", "Austria", "Jordan"): "2026-06-17",
    ("K", "Portugal", "Congo DR"): "2026-06-17",
    ("K", "Uzbekistan", "Colombia"): "2026-06-18",
    ("L", "England", "Croatia"): "2026-06-17",
    ("L", "Ghana", "Panama"): "2026-06-17",
}

# Estimated dates for unplayed matches (MD3 approximate)
MD3_ESTIMATED_DATE = "2026-06-25"
MD2_ESTIMATED_DATE = "2026-06-21"


def get_all_group_fixtures() -> list[dict]:
    """
    Return all 72 group-stage fixtures (all C(4,2) pairs per group).
    Each dict: {match_id, group, team_a, team_b, date, matchday}
    """
    fixtures = []
    for group, teams in WC2026_GROUPS.items():
        for team_a, team_b in combinations(teams, 2):
            # Look up date — check both orderings
            date = (
                KNOWN_MATCH_DATES.get((group, team_a, team_b))
                or KNOWN_MATCH_DATES.get((group, team_b, team_a))
                or MD2_ESTIMATED_DATE  # placeholder
            )
            match_id = f"{group}_{team_a.replace(' ','_')}_{team_b.replace(' ','_')}"
            fixtures.append({
                "match_id": match_id,
                "group": group,
                "team_a": team_a,
                "team_b": team_b,
                "date": date,
            })
    return fixtures
