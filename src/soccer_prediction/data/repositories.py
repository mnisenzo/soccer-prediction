"""Data access layer — query functions returning DataFrames and domain objects."""
from __future__ import annotations

from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from .schema import EloHistory, GroupMembership, Match, MatchPrediction, Team, Tournament, TournamentGroup


# ------------------------------------------------------------------
# Teams
# ------------------------------------------------------------------

def get_teams(session: Session) -> pd.DataFrame:
    rows = session.query(Team).order_by(Team.elo_rating.desc()).all()
    return pd.DataFrame([
        {
            "id": t.id, "name": t.name, "fifa_code": t.fifa_code,
            "confederation": t.confederation, "fifa_ranking": t.fifa_ranking,
            "elo_rating": t.elo_rating,
        }
        for t in rows
    ])


def get_team_by_name(session: Session, name: str) -> Optional[Team]:
    return session.query(Team).filter_by(name=name).first()


def update_elo(session: Session, team_name: str, new_elo: float) -> None:
    team = get_team_by_name(session, team_name)
    if team:
        team.elo_rating = new_elo


# ------------------------------------------------------------------
# Matches
# ------------------------------------------------------------------

def get_completed_matches(session: Session, tournament_name: Optional[str] = None) -> pd.DataFrame:
    q = (
        session.query(Match, Team, Team)
        .join(Team, Match.home_team_id == Team.id)
        .join(Team, Match.away_team_id == Team.id, isouter=True)
        .filter(Match.status == "completed")
    )
    if tournament_name:
        q = q.join(Tournament).filter(Tournament.name == tournament_name)

    rows = session.query(Match).filter(Match.status == "completed").all()
    records = []
    for m in rows:
        records.append({
            "match_id": m.id,
            "tournament_id": m.tournament_id,
            "home_team": m.home_team.name,
            "away_team": m.away_team.name,
            "home_team_id": m.home_team_id,
            "away_team_id": m.away_team_id,
            "match_date": m.match_date,
            "stage": m.stage,
            "group_name": m.group_name,
            "home_goals": m.home_goals,
            "away_goals": m.away_goals,
            "is_neutral": m.is_neutral,
        })
    return pd.DataFrame(records)


def get_fixtures(
    session: Session,
    tournament_name: Optional[str] = None,
    tournament_year: Optional[int] = None,
) -> pd.DataFrame:
    q = session.query(Match).filter(Match.status == "scheduled")
    if tournament_name or tournament_year:
        q = q.join(Tournament)
        if tournament_name:
            q = q.filter(Tournament.name == tournament_name)
        if tournament_year:
            q = q.filter(Tournament.year == tournament_year)

    rows = q.order_by(Match.match_date).all()
    records = []
    for m in rows:
        records.append({
            "match_id": m.id,
            "tournament_id": m.tournament_id,
            "home_team": m.home_team.name,
            "away_team": m.away_team.name,
            "home_team_id": m.home_team_id,
            "away_team_id": m.away_team_id,
            "match_date": m.match_date,
            "stage": m.stage,
            "group_name": m.group_name,
            "is_neutral": m.is_neutral,
        })
    return pd.DataFrame(records)


# ------------------------------------------------------------------
# Tournament groups
# ------------------------------------------------------------------

def get_group_assignments(
    session: Session,
    tournament_name: str,
    tournament_year: int,
) -> dict[str, list[str]]:
    """Return {group_name: [team_name, ...]} for the given tournament."""
    tournament = (
        session.query(Tournament)
        .filter_by(name=tournament_name, year=tournament_year)
        .first()
    )
    if tournament is None:
        return {}

    result: dict[str, list[str]] = {}
    for grp in tournament.groups:
        result[grp.group_name] = [m.team.name for m in grp.memberships]
    return result


# ------------------------------------------------------------------
# Predictions
# ------------------------------------------------------------------

def upsert_prediction(session: Session, prediction: MatchPrediction) -> None:
    existing = (
        session.query(MatchPrediction)
        .filter_by(match_id=prediction.match_id, model_name=prediction.model_name)
        .first()
    )
    if existing:
        existing.home_win_prob = prediction.home_win_prob
        existing.draw_prob = prediction.draw_prob
        existing.away_win_prob = prediction.away_win_prob
        existing.home_goals_exp = prediction.home_goals_exp
        existing.away_goals_exp = prediction.away_goals_exp
        existing.features_snapshot = prediction.features_snapshot
    else:
        session.add(prediction)


def get_predictions(session: Session, model_name: Optional[str] = None) -> pd.DataFrame:
    q = session.query(MatchPrediction)
    if model_name:
        q = q.filter_by(model_name=model_name)
    rows = q.all()
    records = []
    for p in rows:
        m = p.match
        records.append({
            "prediction_id": p.id,
            "match_id": p.match_id,
            "home_team": m.home_team.name,
            "away_team": m.away_team.name,
            "stage": m.stage,
            "group_name": m.group_name,
            "model_name": p.model_name,
            "home_win_prob": p.home_win_prob,
            "draw_prob": p.draw_prob,
            "away_win_prob": p.away_win_prob,
            "home_goals_exp": p.home_goals_exp,
            "away_goals_exp": p.away_goals_exp,
            "created_at": p.created_at,
        })
    return pd.DataFrame(records)
