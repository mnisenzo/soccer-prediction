"""Data loaders: ingest CSVs into the database."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .database import get_session
from .schema import (
    GroupMembership, Match, Team, Tournament, TournamentGroup,
)

log = logging.getLogger(__name__)


class DataLoader:
    """Loads CSV data files into the SQLite database."""

    def __init__(self, db_url: Optional[str] = None) -> None:
        self.db_url = db_url

    # ------------------------------------------------------------------
    # Teams
    # ------------------------------------------------------------------

    def load_teams(self, csv_path: Path) -> int:
        """Load teams from CSV. Columns: name, fifa_code, confederation, fifa_ranking, elo_rating."""
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        count = 0
        with get_session(self.db_url) as session:
            for _, row in df.iterrows():
                existing = session.query(Team).filter_by(name=row["name"]).first()
                if existing:
                    existing.fifa_code = row.get("fifa_code")
                    existing.confederation = row.get("confederation")
                    existing.fifa_ranking = int(row["fifa_ranking"]) if pd.notna(row.get("fifa_ranking")) else None
                    existing.elo_rating = float(row.get("elo_rating", 1500))
                else:
                    session.add(Team(
                        name=row["name"],
                        fifa_code=row.get("fifa_code"),
                        confederation=row.get("confederation"),
                        fifa_ranking=int(row["fifa_ranking"]) if pd.notna(row.get("fifa_ranking")) else None,
                        elo_rating=float(row.get("elo_rating", 1500)),
                    ))
                    count += 1
        log.info("Loaded %d new teams from %s", count, csv_path)
        return count

    # ------------------------------------------------------------------
    # Historical matches
    # ------------------------------------------------------------------

    def load_historical_matches(self, csv_path: Path, tournament_name: str = "Historical") -> int:
        """
        Load completed matches from CSV.
        Columns: home_team, away_team, match_date, home_goals, away_goals,
                 tournament (optional), is_neutral (optional), stage (optional)
        """
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        count = 0
        with get_session(self.db_url) as session:
            tournament_cache: dict[str, Tournament] = {}

            for _, row in df.iterrows():
                home = session.query(Team).filter_by(name=row["home_team"]).first()
                away = session.query(Team).filter_by(name=row["away_team"]).first()
                if home is None or away is None:
                    log.warning("Unknown team in row: %s vs %s", row["home_team"], row["away_team"])
                    continue

                t_name = str(row.get("tournament", tournament_name))
                if t_name not in tournament_cache:
                    t = session.query(Tournament).filter_by(name=t_name).first()
                    if t is None:
                        t = Tournament(name=t_name)
                        session.add(t)
                        session.flush()
                    tournament_cache[t_name] = t
                tournament = tournament_cache[t_name]

                match_date = None
                if pd.notna(row.get("match_date")):
                    try:
                        match_date = pd.to_datetime(row["match_date"]).to_pydatetime()
                    except Exception:
                        pass

                match = Match(
                    tournament_id=tournament.id,
                    home_team_id=home.id,
                    away_team_id=away.id,
                    match_date=match_date,
                    stage=str(row.get("stage", "friendly")),
                    home_goals=int(row["home_goals"]) if pd.notna(row.get("home_goals")) else None,
                    away_goals=int(row["away_goals"]) if pd.notna(row.get("away_goals")) else None,
                    is_neutral=bool(row.get("is_neutral", False)),
                    status="completed",
                )
                session.add(match)
                count += 1

        log.info("Loaded %d historical matches from %s", count, csv_path)
        return count

    # ------------------------------------------------------------------
    # Tournament fixtures
    # ------------------------------------------------------------------

    def load_fixtures(
        self,
        csv_path: Path,
        tournament_name: str,
        tournament_year: int,
    ) -> int:
        """
        Load upcoming fixtures from CSV.
        Columns: home_team, away_team, match_date (optional), stage, group_name (optional), is_neutral
        """
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        count = 0

        with get_session(self.db_url) as session:
            tournament = session.query(Tournament).filter_by(
                name=tournament_name, year=tournament_year
            ).first()
            if tournament is None:
                tournament = Tournament(name=tournament_name, year=tournament_year)
                session.add(tournament)
                session.flush()

            group_cache: dict[str, TournamentGroup] = {}

            for _, row in df.iterrows():
                home = session.query(Team).filter_by(name=row["home_team"]).first()
                away = session.query(Team).filter_by(name=row["away_team"]).first()
                if home is None or away is None:
                    log.warning("Unknown team in fixture: %s vs %s", row["home_team"], row["away_team"])
                    continue

                match_date = None
                if pd.notna(row.get("match_date")):
                    try:
                        match_date = pd.to_datetime(row["match_date"]).to_pydatetime()
                    except Exception:
                        pass

                group_name = str(row["group_name"]) if pd.notna(row.get("group_name")) else None
                if group_name:
                    if group_name not in group_cache:
                        grp = session.query(TournamentGroup).filter_by(
                            tournament_id=tournament.id, group_name=group_name
                        ).first()
                        if grp is None:
                            grp = TournamentGroup(tournament_id=tournament.id, group_name=group_name)
                            session.add(grp)
                            session.flush()
                        group_cache[group_name] = grp
                    grp = group_cache[group_name]

                    # Register group memberships
                    for team in (home, away):
                        exists = session.query(GroupMembership).filter_by(
                            group_id=grp.id, team_id=team.id
                        ).first()
                        if not exists:
                            session.add(GroupMembership(group_id=grp.id, team_id=team.id))

                match = Match(
                    tournament_id=tournament.id,
                    home_team_id=home.id,
                    away_team_id=away.id,
                    match_date=match_date,
                    stage=str(row.get("stage", "group")),
                    group_name=group_name,
                    is_neutral=bool(row.get("is_neutral", True)),
                    status="scheduled",
                )
                session.add(match)
                count += 1

        log.info("Loaded %d fixtures for %s %d", count, tournament_name, tournament_year)
        return count
