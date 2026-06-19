"""Tournament data structures for World Cup 2026 format."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


STAGES = [
    "group",
    "round_of_32",
    "round_of_16",
    "quarterfinal",
    "semifinal",
    "final",
]

STAGE_DISPLAY = {
    "group": "Group Stage",
    "round_of_32": "Round of 32",
    "round_of_16": "Round of 16",
    "quarterfinal": "Quarterfinal",
    "semifinal": "Semifinal",
    "final": "Final",
}


@dataclass
class GroupStanding:
    team: str
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    gf: int = 0
    ga: int = 0

    @property
    def points(self) -> int:
        return self.wins * 3 + self.draws

    @property
    def gd(self) -> int:
        return self.gf - self.ga

    def tiebreak_key(self) -> tuple:
        """Higher is better. Used for sorting within a group."""
        return (self.points, self.gd, self.gf)

    def to_dict(self) -> dict:
        return {
            "team": self.team,
            "played": self.played,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "gf": self.gf,
            "ga": self.ga,
            "gd": self.gd,
            "points": self.points,
        }


@dataclass
class SimulatedMatch:
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    stage: str
    group_name: Optional[str] = None

    @property
    def winner(self) -> Optional[str]:
        if self.home_goals > self.away_goals:
            return self.home_team
        if self.away_goals > self.home_goals:
            return self.away_team
        return None  # draw (only valid in group stage)

    @property
    def result(self) -> str:
        if self.home_goals > self.away_goals:
            return "H"
        if self.home_goals < self.away_goals:
            return "A"
        return "D"


@dataclass
class TournamentConfig:
    """Describes the structure of a World Cup tournament."""

    name: str
    year: int
    # {group_name: [team_name, ...]}
    groups: dict[str, list[str]]
    # Number of third-place teams that advance (WC 2026 = 8)
    third_place_advances: int = 8
    # Knockout bracket structure: round name → number of matches
    # Derived automatically from n_groups
    knockout_rounds: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.knockout_rounds:
            n_adv = sum(2 for _ in self.groups) + self.third_place_advances
            stages = []
            while n_adv > 1:
                if n_adv == 32:
                    stages.append("round_of_32")
                elif n_adv == 16:
                    stages.append("round_of_16")
                elif n_adv == 8:
                    stages.append("quarterfinal")
                elif n_adv == 4:
                    stages.append("semifinal")
                elif n_adv == 2:
                    stages.append("final")
                else:
                    stages.append(f"round_of_{n_adv}")
                n_adv //= 2
            self.knockout_rounds = stages

    @property
    def all_teams(self) -> list[str]:
        return [t for teams in self.groups.values() for t in teams]

    @classmethod
    def world_cup_2026(cls, groups: dict[str, list[str]]) -> "TournamentConfig":
        """WC 2026: 12 groups of 4, top-2 + best-8-third advance to Round of 32."""
        return cls(
            name="FIFA World Cup",
            year=2026,
            groups=groups,
            third_place_advances=8,
        )


# Default WC 2026 groups (placeholder — loaded from CSV/DB at runtime)
DEFAULT_WC2026_GROUPS: dict[str, list[str]] = {
    "A": ["USA", "Germany", "Algeria", "South Korea"],
    "B": ["Brazil", "Switzerland", "Ivory Coast", "Australia"],
    "C": ["France", "Japan", "Ecuador", "New Zealand"],
    "D": ["Spain", "Nigeria", "Croatia", "Iraq"],
    "E": ["England", "Denmark", "Senegal", "Jamaica"],
    "F": ["Argentina", "Poland", "Cameroon", "UAE"],
    "G": ["Portugal", "Serbia", "Colombia", "Saudi Arabia"],
    "H": ["Netherlands", "Hungary", "Morocco", "Chile"],
    "I": ["Belgium", "Italy", "Tunisia", "Canada"],
    "J": ["Mexico", "Czechia", "Ghana", "Iran"],
    "K": ["Uruguay", "Slovakia", "Egypt", "Qatar"],
    "L": ["Costa Rica", "Slovenia", "Panama", "Scotland"],
}
