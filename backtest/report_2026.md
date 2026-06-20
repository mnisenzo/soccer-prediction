# WC 2026 Backtest Report
*Generated 2026-06-20 16:05 UTC*

**Matches evaluated:** 0
> **Small sample warning**: 0 matches. Bootstrap CIs are wide — interpret with caution.

## Summary Metrics

| Model | RPS (lower=better) | 95% CI | Log-Loss | Accuracy |
|-------|-------------------|--------|----------|----------|

*RPS = Ranked Probability Score (lower is better). Historical baseline: 45% W / 27% D / 28% L.*

## Calibration

Calibration analysis requires binning probabilities. With only 0 matches, each bucket has very few samples — formal calibration testing is deferred until 30+ matches are available.

## Upset Detection

**14 upset(s)** out of 32 matches (model favourite was wrong).

| Match | Score | Actual | Model Predicted | Model p(actual) | RPS |
|-------|-------|--------|----------------|-----------------|-----|
| USA vs Australia | 2-0 | A wins | B wins | 17% | 0.502 |
| Spain vs Cabo Verde | 0-0 | Draw | A wins | 2% | 0.447 |
| Sweden vs Tunisia | 5-1 | A wins | B wins | 17% | 0.437 |
| Ivory Coast vs Ecuador | 1-0 | A wins | Draw | 12% | 0.420 |
| USA vs Paraguay | 4-1 | A wins | B wins | 21% | 0.420 |
| Mexico vs South Korea | 1-0 | A wins | B wins | 23% | 0.384 |
| Switzerland vs Qatar | 1-1 | Draw | A wins | 12% | 0.320 |
| Paraguay vs Turkey | 1-0 | A wins | B wins | 31% | 0.302 |
| Portugal vs Congo DR | 1-1 | Draw | A wins | 22% | 0.178 |
| South Africa vs Czechia | 1-1 | Draw | B wins | 27% | 0.145 |
| Iran vs New Zealand | 2-2 | Draw | A wins | 30% | 0.143 |
| Netherlands vs Japan | 2-2 | Draw | B wins | 30% | 0.138 |
| Canada vs Bosnia and H | 1-1 | Draw | A wins | 29% | 0.132 |
| Saudi Arabia vs Uruguay | 1-1 | Draw | B wins | 34% | 0.123 |

### Biggest Surprises (highest RPS = model was most wrong)

- **USA vs Australia** (D): Score 2-0, A wins. Model gave 17% — WRONG. RPS=0.502
- **Spain vs Cabo Verde** (H): Score 0-0, Draw. Model gave 2% — WRONG. RPS=0.447
- **Sweden vs Tunisia** (F): Score 5-1, A wins. Model gave 17% — WRONG. RPS=0.437
- **Ivory Coast vs Ecuador** (E): Score 1-0, A wins. Model gave 12% — WRONG. RPS=0.420
- **USA vs Paraguay** (D): Score 4-1, A wins. Model gave 21% — WRONG. RPS=0.420

### Best Predictions (lowest RPS = model was most confident and correct)

- **Brazil vs Haiti** (C): Score 3-0, A wins. p=83%. RPS=0.018
- **Argentina vs Algeria** (J): Score 3-0, A wins. p=81%. RPS=0.022
- **Haiti vs Scotland** (C): Score 0-1, B wins. p=74%. RPS=0.043
- **Bosnia and Herzegovina vs Switzerland** (B): Score 1-4, B wins. p=71%. RPS=0.046
- **Germany vs Curacao** (E): Score 7-1, A wins. p=67%. RPS=0.063

## Match-by-Match Results

| Date | Group | Match | Score | Actual | Model | Ens p(A) | p(D) | p(B) | p(actual) | RPS |
|------|-------|-------|-------|--------|-------|----------|------|------|-----------|-----|
| 2026-06-11 | A | Mexico v South Africa | 2-0 | A wins | ✓ A wins | 49% | 34% | 17% | 49% | 0.147 |
| 2026-06-12 | B | Canada v Bosnia and Herzegovina | 1-1 | Draw | ✗ A wins | 43% | 29% | 28% | 29% | 0.132 |
| 2026-06-12 | A | South Korea v Czechia | 2-1 | A wins | ✓ A wins | 64% | 20% | 16% | 64% | 0.079 |
| 2026-06-13 | D | USA v Paraguay | 4-1 | A wins | ✗ B wins | 21% | 33% | 46% | 21% | 0.420 |
| 2026-06-13 | B | Switzerland v Qatar | 1-1 | Draw | ✗ A wins | 80% | 12% | 9% | 12% | 0.320 |
| 2026-06-13 | C | Brazil v Morocco | 1-1 | Draw | ✓ Draw | 28% | 38% | 34% | 38% | 0.098 |
| 2026-06-14 | E | Germany v Curacao | 7-1 | A wins | ✓ A wins | 67% | 22% | 12% | 67% | 0.063 |
| 2026-06-14 | E | Ivory Coast v Ecuador | 1-0 | A wins | ✗ Draw | 12% | 61% | 27% | 12% | 0.420 |
| 2026-06-14 | C | Haiti v Scotland | 0-1 | B wins | ✓ B wins | 13% | 13% | 74% | 74% | 0.043 |
| 2026-06-14 | F | Netherlands v Japan | 2-2 | Draw | ✗ B wins | 23% | 30% | 47% | 30% | 0.138 |
| 2026-06-14 | D | Australia v Turkey | 2-0 | A wins | ✓ A wins | 40% | 29% | 31% | 40% | 0.225 |
| 2026-06-15 | H | Spain v Cabo Verde | 0-0 | Draw | ✗ A wins | 95% | 2% | 3% | 2% | 0.447 |
| 2026-06-15 | F | Sweden v Tunisia | 5-1 | A wins | ✗ B wins | 17% | 39% | 43% | 17% | 0.437 |
| 2026-06-15 | G | Belgium v Egypt | 1-1 | Draw | ✓ Draw | 27% | 42% | 31% | 42% | 0.085 |
| 2026-06-15 | H | Saudi Arabia v Uruguay | 1-1 | Draw | ✗ B wins | 21% | 34% | 45% | 34% | 0.123 |
| 2026-06-16 | G | Iran v New Zealand | 2-2 | Draw | ✗ A wins | 49% | 30% | 21% | 30% | 0.143 |
| 2026-06-16 | I | France v Senegal | 3-1 | A wins | ✓ A wins | 42% | 27% | 31% | 42% | 0.220 |
| 2026-06-16 | I | Norway v Iraq | 4-1 | A wins | ✓ A wins | 57% | 25% | 18% | 57% | 0.108 |
| 2026-06-17 | L | England v Croatia | 4-2 | A wins | ✓ A wins | 56% | 29% | 14% | 56% | 0.106 |
| 2026-06-17 | K | Portugal v Congo DR | 1-1 | Draw | ✗ A wins | 55% | 22% | 22% | 22% | 0.178 |
| 2026-06-17 | J | Argentina v Algeria | 3-0 | A wins | ✓ A wins | 81% | 11% | 8% | 81% | 0.022 |
| 2026-06-17 | L | Ghana v Panama | 1-0 | A wins | ✓ A wins | 39% | 31% | 29% | 39% | 0.226 |
| 2026-06-17 | J | Austria v Jordan | 3-1 | A wins | ✓ A wins | 35% | 32% | 33% | 35% | 0.263 |
| 2026-06-18 | B | Canada v Qatar | 6-0 | A wins | ✓ A wins | 49% | 24% | 27% | 49% | 0.169 |
| 2026-06-18 | K | Uzbekistan v Colombia | 1-3 | B wins | ✓ B wins | 29% | 29% | 42% | 42% | 0.214 |
| 2026-06-18 | B | Bosnia and Herzegovina v Switzerland | 1-4 | B wins | ✓ B wins | 10% | 18% | 71% | 71% | 0.046 |
| 2026-06-18 | A | South Africa v Czechia | 1-1 | Draw | ✗ B wins | 25% | 27% | 47% | 27% | 0.145 |
| 2026-06-19 | D | USA v Australia | 2-0 | A wins | ✗ B wins | 17% | 26% | 57% | 17% | 0.502 |
| 2026-06-19 | C | Morocco v Scotland | 1-0 | A wins | ✓ A wins | 49% | 30% | 21% | 49% | 0.153 |
| 2026-06-19 | A | Mexico v South Korea | 1-0 | A wins | ✗ B wins | 23% | 36% | 41% | 23% | 0.384 |
| 2026-06-20 | D | Paraguay v Turkey | 1-0 | A wins | ✗ B wins | 31% | 32% | 37% | 31% | 0.302 |
| 2026-06-20 | C | Brazil v Haiti | 3-0 | A wins | ✓ A wins | 83% | 10% | 8% | 83% | 0.018 |

## Kalshi Market Comparison

Pre-match Kalshi odds not available for completed matches. Settled markets reset to 0.99/0.01 post-match; Wayback Machine snapshots did not yield price data.

**Note**: Kalshi winner market live odds (retrieved June 19, 2026): France 17.5%, Spain 12.4%, England 12.1%, Argentina 10.1%. Model disagrees most on Mexico (+14.9pp) and France (-14.1pp).

## Methodology

- **RPS** (Ranked Probability Score): primary metric. Formula: `0.5 * [(F1-O1)² + (F1+F2 - O1-O2)²]`
- **Ensemble**: Dixon-Coles 66.7% + Elo 33.3% (XGBoost not trained in this run)
- **Predictions frozen pre-tournament** — immutable, not updated as matches complete
- **Historical baseline**: 45%/27%/28% W/D/L from 2010-2022 WC group stages
- **Bootstrap CI**: 10,000 resamples with seed=42
