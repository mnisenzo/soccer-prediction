# WC 2026 Backtest Report
*Generated 2026-06-20 14:56 UTC*

**Matches evaluated:** 32
> **Small sample warning**: 32 matches. Bootstrap CIs are wide — interpret with caution.

## Summary Metrics

| Model | RPS (lower=better) | 95% CI | Log-Loss | Accuracy |
|-------|-------------------|--------|----------|----------|
| Ensemble **[beats baseline]** | 0.1334 | [0.106, 0.162] | 0.8224 | 65.6% |
| Dixon-Coles **[beats baseline]** | 0.1203 | [0.091, 0.152] | 0.7572 | 65.6% |
| Elo | 0.1804 | [0.148, 0.213] | 1.0248 | 50.0% |
| Historical WC baseline | 0.1908 | [0.172, 0.213] | 1.0026 | 59.4% |
| Uniform (1/3 each) | 0.2257 | [0.200, 0.252] | 1.0986 | 40.6% |

*RPS = Ranked Probability Score (lower is better). Historical baseline: 45% W / 27% D / 28% L.*

## Dixon-Coles Expected Goals

| Metric | Value |
|--------|-------|
| Goals-for MAE | 0.96 |
| Goals-against MAE | 0.46 |
| Correct goal-scorer direction | 26/32 (81%) |

## Calibration

Calibration analysis requires binning probabilities. With only 32 matches, each bucket has very few samples — formal calibration testing is deferred until 30+ matches are available.

## Upset Detection

**11 upset(s)** out of 32 matches (model favourite was wrong).

| Match | Score | Actual | Model Predicted | Model p(actual) | RPS |
|-------|-------|--------|----------------|-----------------|-----|
| Switzerland vs Qatar | 1-1 | Draw | A wins | 12% | 0.315 |
| Ivory Coast vs Ecuador | 1-0 | A wins | Draw | 24% | 0.309 |
| USA vs Australia | 2-0 | A wins | B wins | 34% | 0.294 |
| Ghana vs Panama | 1-0 | A wins | Draw | 37% | 0.224 |
| Canada vs Bosnia and H | 1-1 | Draw | A wins | 21% | 0.219 |
| Spain vs Cabo Verde | 0-0 | Draw | A wins | 29% | 0.194 |
| Iran vs New Zealand | 2-2 | Draw | B wins | 25% | 0.141 |
| Netherlands vs Japan | 2-2 | Draw | B wins | 28% | 0.139 |
| Portugal vs Congo DR | 1-1 | Draw | A wins | 28% | 0.134 |
| Saudi Arabia vs Uruguay | 1-1 | Draw | B wins | 33% | 0.132 |
| South Africa vs Czechia | 1-1 | Draw | B wins | 31% | 0.120 |

### Biggest Surprises (highest RPS = model was most wrong)

- **Switzerland vs Qatar** (B): Score 1-1, Draw. Model gave 12% — WRONG. RPS=0.315
- **Ivory Coast vs Ecuador** (E): Score 1-0, A wins. Model gave 24% — WRONG. RPS=0.309
- **USA vs Australia** (D): Score 2-0, A wins. Model gave 34% — WRONG. RPS=0.294
- **Paraguay vs Turkey** (D): Score 1-0, A wins. Model gave 36% — correct. RPS=0.257
- **Ghana vs Panama** (L): Score 1-0, A wins. Model gave 37% — WRONG. RPS=0.224

### Best Predictions (lowest RPS = model was most confident and correct)

- **Germany vs Curacao** (E): Score 7-1, A wins. p=85%. RPS=0.013
- **Canada vs Qatar** (B): Score 6-0, A wins. p=78%. RPS=0.028
- **Argentina vs Algeria** (J): Score 3-0, A wins. p=77%. RPS=0.029
- **Brazil vs Haiti** (C): Score 3-0, A wins. p=73%. RPS=0.041
- **Norway vs Iraq** (I): Score 4-1, A wins. p=72%. RPS=0.048

## Match-by-Match Results

| Date | Group | Match | Score | Actual | Model | Ens p(A) | p(D) | p(B) | p(actual) | RPS |
|------|-------|-------|-------|--------|-------|----------|------|------|-----------|-----|
| 2026-06-11 | A | Mexico v South Africa | 2-0 | A wins | ✓ A wins | 69% | 24% | 7% | 69% | 0.050 |
| 2026-06-12 | A | South Korea v Czechia | 2-1 | A wins | ✓ A wins | 58% | 24% | 18% | 58% | 0.106 |
| 2026-06-12 | B | Canada v Bosnia and Herzegovina | 1-1 | Draw | ✗ A wins | 65% | 21% | 14% | 21% | 0.219 |
| 2026-06-13 | B | Switzerland v Qatar | 1-1 | Draw | ✗ A wins | 79% | 12% | 9% | 12% | 0.315 |
| 2026-06-13 | C | Brazil v Morocco | 1-1 | Draw | ✓ Draw | 32% | 38% | 30% | 38% | 0.097 |
| 2026-06-13 | D | USA v Paraguay | 4-1 | A wins | ✓ A wins | 54% | 23% | 23% | 54% | 0.133 |
| 2026-06-14 | C | Haiti v Scotland | 0-1 | B wins | ✓ B wins | 17% | 24% | 60% | 60% | 0.095 |
| 2026-06-14 | F | Netherlands v Japan | 2-2 | Draw | ✗ B wins | 27% | 28% | 46% | 28% | 0.139 |
| 2026-06-14 | E | Germany v Curacao | 7-1 | A wins | ✓ A wins | 85% | 8% | 6% | 85% | 0.013 |
| 2026-06-14 | E | Ivory Coast v Ecuador | 1-0 | A wins | ✗ Draw | 24% | 56% | 20% | 24% | 0.309 |
| 2026-06-14 | D | Australia v Turkey | 2-0 | A wins | ✓ A wins | 50% | 29% | 20% | 50% | 0.145 |
| 2026-06-15 | H | Saudi Arabia v Uruguay | 1-1 | Draw | ✗ B wins | 19% | 33% | 48% | 33% | 0.132 |
| 2026-06-15 | H | Spain v Cabo Verde | 0-0 | Draw | ✗ A wins | 62% | 29% | 10% | 29% | 0.194 |
| 2026-06-15 | F | Sweden v Tunisia | 5-1 | A wins | ✓ A wins | 64% | 20% | 16% | 64% | 0.078 |
| 2026-06-15 | G | Belgium v Egypt | 1-1 | Draw | ✓ Draw | 29% | 36% | 34% | 36% | 0.102 |
| 2026-06-16 | G | Iran v New Zealand | 2-2 | Draw | ✗ B wins | 36% | 25% | 39% | 25% | 0.141 |
| 2026-06-16 | I | Norway v Iraq | 4-1 | A wins | ✓ A wins | 72% | 16% | 12% | 72% | 0.048 |
| 2026-06-16 | I | France v Senegal | 3-1 | A wins | ✓ A wins | 58% | 22% | 20% | 58% | 0.110 |
| 2026-06-17 | J | Argentina v Algeria | 3-0 | A wins | ✓ A wins | 77% | 15% | 8% | 77% | 0.029 |
| 2026-06-17 | J | Austria v Jordan | 3-1 | A wins | ✓ A wins | 52% | 27% | 22% | 52% | 0.141 |
| 2026-06-17 | K | Portugal v Congo DR | 1-1 | Draw | ✗ A wins | 42% | 28% | 30% | 28% | 0.134 |
| 2026-06-17 | L | Ghana v Panama | 1-0 | A wins | ✗ Draw | 37% | 39% | 24% | 37% | 0.224 |
| 2026-06-17 | L | England v Croatia | 4-2 | A wins | ✓ A wins | 60% | 22% | 18% | 60% | 0.099 |
| 2026-06-18 | A | South Africa v Czechia | 1-1 | Draw | ✗ B wins | 34% | 31% | 35% | 31% | 0.120 |
| 2026-06-18 | B | Canada v Qatar | 6-0 | A wins | ✓ A wins | 78% | 12% | 10% | 78% | 0.028 |
| 2026-06-18 | K | Uzbekistan v Colombia | 1-3 | B wins | ✓ B wins | 15% | 20% | 65% | 65% | 0.074 |
| 2026-06-18 | B | Bosnia and Herzegovina v Switzerland | 1-4 | B wins | ✓ B wins | 14% | 21% | 65% | 65% | 0.072 |
| 2026-06-19 | D | USA v Australia | 2-0 | A wins | ✗ B wins | 34% | 28% | 38% | 34% | 0.294 |
| 2026-06-19 | A | Mexico v South Korea | 1-0 | A wins | ✓ A wins | 50% | 36% | 13% | 50% | 0.133 |
| 2026-06-19 | C | Morocco v Scotland | 1-0 | A wins | ✓ A wins | 42% | 35% | 23% | 42% | 0.197 |
| 2026-06-20 | D | Paraguay v Turkey | 1-0 | A wins | ✓ A wins | 36% | 31% | 33% | 36% | 0.257 |
| 2026-06-20 | C | Brazil v Haiti | 3-0 | A wins | ✓ A wins | 73% | 17% | 10% | 73% | 0.041 |

## Kalshi Market Comparison

Pre-match Kalshi odds not available for completed matches. Settled markets reset to 0.99/0.01 post-match; Wayback Machine snapshots did not yield price data.

**Note**: Kalshi winner market live odds (retrieved June 19, 2026): France 17.5%, Spain 12.4%, England 12.1%, Argentina 10.1%. Model disagrees most on Mexico (+14.9pp) and France (-14.1pp).

## Methodology

- **RPS** (Ranked Probability Score): primary metric. Formula: `0.5 * [(F1-O1)² + (F1+F2 - O1-O2)²]`
- **Ensemble**: Dixon-Coles 66.7% + Elo 33.3% (XGBoost not trained in this run)
- **Predictions frozen pre-tournament** — immutable, not updated as matches complete
- **Historical baseline**: 45%/27%/28% W/D/L from 2010-2022 WC group stages
- **Bootstrap CI**: 10,000 resamples with seed=42
