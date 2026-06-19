"""Shared pytest fixtures."""
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from soccer_prediction.data.database import init_db, reset_engine, get_session
from soccer_prediction.data.schema import Match, Team, Tournament


TEAMS = ["Brazil", "France", "Argentina", "England", "Germany",
         "Spain", "Netherlands", "Portugal", "Italy", "USA"]

ELO_RATINGS = {
    "Brazil": 2085, "France": 2065, "Argentina": 2060,
    "England": 2040, "Germany": 1985, "Spain": 2010,
    "Netherlands": 2020, "Portugal": 2030, "Italy": 2005, "USA": 1975,
}


@pytest.fixture
def tmp_db(tmp_path):
    """Provide a fresh temporary SQLite database for each test."""
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"
    reset_engine()
    init_db(db_url)
    yield db_url
    reset_engine()


@pytest.fixture
def sample_matches() -> pd.DataFrame:
    """Small DataFrame of completed matches for model fitting."""
    records = [
        {"home_team": "Brazil", "away_team": "Argentina", "match_date": "2023-01-01",
         "home_goals": 2, "away_goals": 1, "is_neutral": True, "stage": "friendly"},
        {"home_team": "France", "away_team": "England", "match_date": "2023-02-01",
         "home_goals": 1, "away_goals": 1, "is_neutral": True, "stage": "friendly"},
        {"home_team": "Germany", "away_team": "Spain", "match_date": "2023-03-01",
         "home_goals": 0, "away_goals": 2, "is_neutral": True, "stage": "friendly"},
        {"home_team": "Netherlands", "away_team": "Portugal", "match_date": "2023-04-01",
         "home_goals": 3, "away_goals": 1, "is_neutral": False, "stage": "friendly"},
        {"home_team": "Italy", "away_team": "USA", "match_date": "2023-05-01",
         "home_goals": 2, "away_goals": 0, "is_neutral": True, "stage": "friendly"},
        {"home_team": "Argentina", "away_team": "France", "match_date": "2023-06-01",
         "home_goals": 1, "away_goals": 0, "is_neutral": True, "stage": "friendly"},
        {"home_team": "England", "away_team": "Germany", "match_date": "2023-07-01",
         "home_goals": 2, "away_goals": 2, "is_neutral": True, "stage": "friendly"},
        {"home_team": "Spain", "away_team": "Brazil", "match_date": "2023-08-01",
         "home_goals": 1, "away_goals": 3, "is_neutral": True, "stage": "friendly"},
        {"home_team": "USA", "away_team": "Netherlands", "match_date": "2023-09-01",
         "home_goals": 0, "away_goals": 1, "is_neutral": False, "stage": "friendly"},
        {"home_team": "Portugal", "away_team": "Italy", "match_date": "2023-10-01",
         "home_goals": 2, "away_goals": 1, "is_neutral": True, "stage": "friendly"},
    ]
    return pd.DataFrame(records)


@pytest.fixture
def simple_groups() -> dict:
    return {
        "A": ["Brazil", "France", "Germany", "USA"],
        "B": ["Argentina", "England", "Spain", "Italy"],
        "C": ["Netherlands", "Portugal", "Argentina", "USA"],  # overlapping names OK for testing
    }
