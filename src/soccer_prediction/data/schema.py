"""SQLAlchemy ORM schema for the soccer prediction system."""
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON,
    String, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, relationship


class Base(DeclarativeBase):
    pass


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = Column(Integer, primary_key=True)
    name: Mapped[str] = Column(String, nullable=False, unique=True)
    fifa_code: Mapped[Optional[str]] = Column(String(3))
    confederation: Mapped[Optional[str]] = Column(String)  # UEFA, CONMEBOL, CONCACAF, CAF, AFC, OFC
    fifa_ranking: Mapped[Optional[int]] = Column(Integer)
    elo_rating: Mapped[float] = Column(Float, default=1500.0)

    home_matches = relationship("Match", foreign_keys="Match.home_team_id", back_populates="home_team")
    away_matches = relationship("Match", foreign_keys="Match.away_team_id", back_populates="away_team")

    def __repr__(self) -> str:
        return f"<Team {self.name} ({self.fifa_code}) elo={self.elo_rating:.0f}>"


class Tournament(Base):
    __tablename__ = "tournaments"

    id: Mapped[int] = Column(Integer, primary_key=True)
    name: Mapped[str] = Column(String, nullable=False)
    year: Mapped[Optional[int]] = Column(Integer)
    host_country: Mapped[Optional[str]] = Column(String)
    start_date: Mapped[Optional[datetime]] = Column(DateTime)
    end_date: Mapped[Optional[datetime]] = Column(DateTime)

    matches = relationship("Match", back_populates="tournament")
    groups = relationship("TournamentGroup", back_populates="tournament")

    __table_args__ = (UniqueConstraint("name", "year"),)


class TournamentGroup(Base):
    __tablename__ = "tournament_groups"

    id: Mapped[int] = Column(Integer, primary_key=True)
    tournament_id: Mapped[int] = Column(Integer, ForeignKey("tournaments.id"))
    group_name: Mapped[str] = Column(String)  # A, B, C, ...

    tournament = relationship("Tournament", back_populates="groups")
    memberships = relationship("GroupMembership", back_populates="group")

    __table_args__ = (UniqueConstraint("tournament_id", "group_name"),)


class GroupMembership(Base):
    __tablename__ = "group_memberships"

    id: Mapped[int] = Column(Integer, primary_key=True)
    group_id: Mapped[int] = Column(Integer, ForeignKey("tournament_groups.id"))
    team_id: Mapped[int] = Column(Integer, ForeignKey("teams.id"))

    group = relationship("TournamentGroup", back_populates="memberships")
    team = relationship("Team")

    __table_args__ = (UniqueConstraint("group_id", "team_id"),)


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[int] = Column(Integer, primary_key=True)
    tournament_id: Mapped[Optional[int]] = Column(Integer, ForeignKey("tournaments.id"), nullable=True)
    home_team_id: Mapped[int] = Column(Integer, ForeignKey("teams.id"))
    away_team_id: Mapped[int] = Column(Integer, ForeignKey("teams.id"))
    match_date: Mapped[Optional[datetime]] = Column(DateTime)
    # "group", "round_of_32", "round_of_16", "quarterfinal", "semifinal", "third_place", "final"
    stage: Mapped[Optional[str]] = Column(String, default="group")
    group_name: Mapped[Optional[str]] = Column(String, nullable=True)
    home_goals: Mapped[Optional[int]] = Column(Integer, nullable=True)
    away_goals: Mapped[Optional[int]] = Column(Integer, nullable=True)
    is_neutral: Mapped[bool] = Column(Boolean, default=True)
    # "scheduled", "completed"
    status: Mapped[str] = Column(String, default="scheduled")

    tournament = relationship("Tournament", back_populates="matches")
    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_matches")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_matches")
    predictions = relationship("MatchPrediction", back_populates="match", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        score = (
            f"{self.home_goals}-{self.away_goals}"
            if self.home_goals is not None else "vs"
        )
        return f"<Match {self.home_team_id} {score} {self.away_team_id}>"


class MatchPrediction(Base):
    __tablename__ = "match_predictions"

    id: Mapped[int] = Column(Integer, primary_key=True)
    match_id: Mapped[int] = Column(Integer, ForeignKey("matches.id"))
    model_name: Mapped[str] = Column(String)
    home_win_prob: Mapped[float] = Column(Float)
    draw_prob: Mapped[float] = Column(Float)
    away_win_prob: Mapped[float] = Column(Float)
    home_goals_exp: Mapped[Optional[float]] = Column(Float, nullable=True)
    away_goals_exp: Mapped[Optional[float]] = Column(Float, nullable=True)
    features_snapshot: Mapped[Optional[dict]] = Column(JSON, nullable=True)
    created_at: Mapped[datetime] = Column(DateTime, default=datetime.utcnow)

    match = relationship("Match", back_populates="predictions")

    __table_args__ = (UniqueConstraint("match_id", "model_name"),)


class EloHistory(Base):
    __tablename__ = "elo_history"

    id: Mapped[int] = Column(Integer, primary_key=True)
    team_id: Mapped[int] = Column(Integer, ForeignKey("teams.id"))
    match_id: Mapped[int] = Column(Integer, ForeignKey("matches.id"))
    elo_before: Mapped[float] = Column(Float)
    elo_after: Mapped[float] = Column(Float)
    recorded_at: Mapped[datetime] = Column(DateTime, default=datetime.utcnow)

    team = relationship("Team")
    match = relationship("Match")
