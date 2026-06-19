# Soccer Prediction System

A modular prediction framework for the FIFA World Cup and Kalshi-style prediction markets.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic historical data
python scripts/generate_sample_data.py

# 3. Load data into the database
python scripts/load_data.py

# 4. Run the end-to-end pipeline (fit model + simulate + evaluate markets)
python scripts/run_pipeline.py --model elo_logistic --n-sims 10000

# 5. Launch the Streamlit app
streamlit run app/streamlit_app.py
```

---

## Project Structure

```
soccer-prediction/
├── src/soccer_prediction/
│   ├── data/           # SQLAlchemy schema, DB init, CSV loaders, repositories
│   ├── features/       # FeatureExtractor ABC, Elo, form, pipeline, registry
│   ├── models/         # SoccerModel ABC, EloLogistic, PoissonGoals, registry
│   ├── simulation/     # Tournament dataclasses, group stage, Monte Carlo
│   ├── markets/        # Market ABC, YAML registry, evaluator
│   └── utils/          # Logging
├── app/
│   └── streamlit_app.py
├── configs/
│   ├── settings.yaml
│   └── markets/
│       ├── world_cup_2026.yaml   # 50+ pre-built markets
│       └── example_markets.yaml  # Template for custom markets
├── data/
│   ├── raw/            # Drop your CSV/API files here
│   ├── processed/      # Feature outputs, model predictions
│   └── sample/         # Toy data (teams, matches, WC 2026 fixtures)
├── scripts/
│   ├── generate_sample_data.py
│   ├── load_data.py
│   └── run_pipeline.py
└── tests/
```

---

## Streamlit App

The app has 6 sections (left sidebar):

| Section | What it does |
|---------|-------------|
| **Overview** | DB stats, top teams by Elo, quick-start guide |
| **Match Predictor** | Select any two teams → win/draw/loss probs, xG, score heatmap, market stats |
| **Tournament Simulation** | Run N Monte Carlo trials → stage-reach probs, group winner probs, heatmap |
| **Market Analysis** | Compare model probs to Kalshi prices, compute edge, export CSV |
| **Feature Inspector** | View Elo ratings, Poisson attack/defense table, match-level feature snapshot |
| **Data Manager** | Upload/view teams, historical matches, fixtures |

---

## How to Add a New Model

1. Create `src/soccer_prediction/models/my_model.py`:

```python
from pathlib import Path
import joblib
import pandas as pd
from .base import SoccerModel, MatchPrediction

class MyModel(SoccerModel):
    name = "my_model"  # unique identifier

    def fit(self, matches: pd.DataFrame) -> "MyModel":
        # learn from matches: columns include home_team, away_team,
        # home_goals, away_goals, is_neutral, match_date, stage
        ...
        return self

    def predict_match(self, home_team, away_team, is_neutral=True, **ctx) -> MatchPrediction:
        return MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            home_win_prob=...,
            draw_prob=...,
            away_win_prob=...,
        )

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "MyModel":
        return joblib.load(path)
```

2. Register it in `src/soccer_prediction/models/registry.py`:

```python
from .my_model import MyModel
_REGISTRY[MyModel.name] = MyModel
```

3. That's it — the model will appear in the Streamlit app dropdown and `run_pipeline.py`.

---

## How to Add a New Kalshi Market

### Option 1: Edit a YAML config (no code required)

Add an entry to `configs/markets/world_cup_2026.yaml` (or create a new `.yaml` file):

```yaml
markets:
  - id: my_market_id
    name: "Netherlands wins World Cup"
    category: tournament_winner
    params:
      team: Netherlands
    market_price: 0.06   # Kalshi implied probability
```

The app and `run_pipeline.py` automatically load all `*.yaml` files from `configs/markets/`.

### Option 2: Add a new market category

1. Add the category name to `MARKET_CATEGORIES` in `src/soccer_prediction/markets/base.py`.
2. Implement it in the `_dispatch` method in `src/soccer_prediction/markets/evaluator.py`.
3. Define markets of that type in YAML.

### Supported categories

| Category | Required params |
|----------|----------------|
| `tournament_winner` | `team` |
| `reaches_stage` | `team`, `stage` (round_of_32, round_of_16, quarterfinal, semifinal, final) |
| `advances_from_group` | `team` |
| `group_winner` | `team`, `group` |
| `match_outcome` | `home_team`, `away_team`, `outcome` (H/D/A), `is_neutral` |
| `over_under_goals` | `home_team`, `away_team`, `threshold`, `direction` (over/under), `is_neutral` |
| `both_teams_score` | `home_team`, `away_team`, `is_neutral` |
| `clean_sheet` | `team`, `home_team`, `away_team`, `is_neutral` |
| `correct_score` | `home_team`, `away_team`, `home_goals`, `away_goals`, `is_neutral` |

---

## Models

### EloLogistic (`elo_logistic`)

Uses Elo rating differences to predict 3-way outcomes:

```
E_home = 1 / (1 + 10^(-(elo_home - elo_away) / 400))
P(draw) = 0.28 × (1 - |2·E_home - 1|)   # highest when evenly matched
P(home_win) = E_home × (1 - P(draw))
P(away_win) = (1 - E_home) × (1 - P(draw))
```

Elo ratings are updated from match history with K=32, home advantage=100 Elo points.

### PoissonGoals (`poisson_goals`)

Models goals as independent Poisson processes:

```
λ_home = attack[home] × defense[away] × avg_goals
λ_away = attack[away] × defense[home] × avg_goals
P(score = i:j) = Poisson(i; λ_home) × Poisson(j; λ_away)
```

Attack and defense parameters are normalised team averages. Provides score matrices for richer market evaluation (over/under, BTTS, correct score, clean sheet).

---

## Data Layer

- **Database**: SQLite by default. Swap in `configs/settings.yaml`:
  ```yaml
  database:
    url: "postgresql://user:pass@localhost/soccer_prediction"
  ```
- **Schema tables**: `teams`, `tournaments`, `tournament_groups`, `group_memberships`, `matches`, `match_predictions`, `elo_history`
- **CSV columns for historical matches**: `home_team`, `away_team`, `match_date`, `home_goals`, `away_goals`, `tournament` (opt), `is_neutral` (opt), `stage` (opt)
- **CSV columns for fixtures**: `home_team`, `away_team`, `match_date`, `stage`, `group_name`, `is_neutral`

---

## Running Tests

```bash
pytest tests/ -v
# With coverage:
pytest tests/ -v --cov=soccer_prediction --cov-report=term-missing
```

---

## Tournament Format (WC 2026)

- 48 teams in 12 groups of 4
- Top 2 from each group advance (24 teams)
- Best 8 third-place teams advance (32 total → Round of 32)
- 4 knockout rounds: Round of 32 → Round of 16 → Quarterfinal → Semifinal → Final
- Tiebreakers: Points → Goal Difference → Goals For → H2H → Random lot
