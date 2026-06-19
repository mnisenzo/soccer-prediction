"""Load and manage market definitions from YAML config files."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

from .base import MarketDefinition

log = logging.getLogger(__name__)


class MarketRegistry:
    """
    Loads market definitions from YAML files and stores them by ID.

    YAML format::

        markets:
          - id: arg_wins_wc
            name: "Argentina wins World Cup"
            category: tournament_winner
            params:
              team: Argentina
            market_price: 0.18   # optional Kalshi price

    To add a new market: add an entry to your YAML config.
    To add a new category: implement it in markets/evaluator.py.
    """

    def __init__(self) -> None:
        self._markets: dict[str, MarketDefinition] = {}

    def load_yaml(self, path: Path) -> int:
        """Load market definitions from a YAML file. Returns number of markets loaded."""
        path = Path(path)
        if not path.exists():
            log.warning("Market config not found: %s", path)
            return 0

        with open(path) as f:
            data = yaml.safe_load(f)

        markets = data.get("markets", [])
        count = 0
        for entry in markets:
            try:
                mkt = MarketDefinition(
                    id=entry["id"],
                    name=entry["name"],
                    category=entry["category"],
                    params=entry.get("params", {}),
                    market_price=entry.get("market_price"),
                )
                self._markets[mkt.id] = mkt
                count += 1
            except Exception as exc:
                log.warning("Skipping market entry %s: %s", entry.get("id"), exc)

        log.info("Loaded %d markets from %s", count, path)
        return count

    def load_directory(self, directory: Path) -> int:
        """Load all *.yaml files from a directory."""
        total = 0
        for yaml_file in Path(directory).glob("*.yaml"):
            total += self.load_yaml(yaml_file)
        return total

    def register(self, market: MarketDefinition) -> None:
        self._markets[market.id] = market

    def get(self, market_id: str) -> Optional[MarketDefinition]:
        return self._markets.get(market_id)

    def all(self) -> list[MarketDefinition]:
        return list(self._markets.values())

    def by_category(self, category: str) -> list[MarketDefinition]:
        return [m for m in self._markets.values() if m.category == category]

    def update_price(self, market_id: str, price: float) -> None:
        """Update the market price for a market (for comparison against model prob)."""
        if market_id in self._markets:
            self._markets[market_id].market_price = price

    def __len__(self) -> int:
        return len(self._markets)

    def __repr__(self) -> str:
        return f"<MarketRegistry {len(self)} markets>"
