"""Market definition data structures."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# ------------------------------------------------------------------
# Supported market categories
# ------------------------------------------------------------------
# Add new categories here and implement them in evaluator.py

MARKET_CATEGORIES = [
    "tournament_winner",        # team wins the tournament
    "reaches_stage",            # team reaches a specific stage
    "advances_from_group",      # team finishes top-2 or best-8-third
    "group_winner",             # team wins their group
    "match_outcome",            # specific match result (H/D/A)
    "over_under_goals",         # total goals over/under threshold
    "both_teams_score",         # both teams score ≥1 goal
    "clean_sheet",              # specified team concedes 0 goals
    "correct_score",            # exact score prediction
    "anytime_winner",           # team wins at any point (alias for reaches_final+)
]


@dataclass
class MarketDefinition:
    """
    Defines a single prediction market.

    Attributes:
        id: unique identifier (slug)
        name: human-readable description
        category: one of MARKET_CATEGORIES
        params: category-specific parameters (see evaluator.py for each category)
        market_price: optional Kalshi/market implied probability (0–1) for comparison
    """

    id: str
    name: str
    category: str
    params: dict[str, Any] = field(default_factory=dict)
    market_price: Optional[float] = None

    def __post_init__(self) -> None:
        if self.category not in MARKET_CATEGORIES:
            raise ValueError(
                f"Unknown market category: '{self.category}'. "
                f"Valid options: {MARKET_CATEGORIES}"
            )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "params": self.params,
            "market_price": self.market_price,
        }


@dataclass
class MarketResult:
    """The evaluated probability for a single market."""

    market: MarketDefinition
    model_probability: float
    market_price: Optional[float] = None  # user-input Kalshi price

    @property
    def edge(self) -> Optional[float]:
        """Model prob minus market implied prob. Positive = model sees value."""
        if self.market_price is not None:
            return round(self.model_probability - self.market_price, 4)
        return None

    def to_dict(self) -> dict:
        return {
            "id": self.market.id,
            "name": self.market.name,
            "category": self.market.category,
            "model_probability": round(self.model_probability, 4),
            "market_price": self.market_price,
            "edge": self.edge,
        }
