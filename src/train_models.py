"""
Train three soccer prediction models:
  A) Dixon-Coles Poisson (time-weighted)
  B) XGBoost classifier (3-way: home win / draw / away win)
  C) Elo baseline (analytical)

Usage:
    python src/train_models.py [--skip-xgb] [--xgb-trials 50]

Outputs:
    models/dixon_coles_params.json
    models/xgb_wc2026.json
    models/elo_ratings.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
TODAY = pd.Timestamp("today")


# ─────────────────────────────────────────────────────────────
# Model A: Dixon-Coles Poisson
# ─────────────────────────────────────────────────────────────

TOURNAMENT_K = {
    "FIFA World Cup": 4.0,
    "UEFA Euro": 2.0,
    "Copa América": 2.0,
    "Africa Cup of Nations": 2.0,
    "Asian Cup": 2.0,
    "CONCACAF Gold Cup": 2.0,
    "FIFA World Cup qualification": 1.5,
    "Friendly": 0.5,
}


def _importance_weight(tournament: str) -> float:
    for key, v in TOURNAMENT_K.items():
        if key.lower() in str(tournament).lower():
            return v
    return 1.0


def _dc_tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles correction for low-scoring matches."""
    if x == 0 and y == 0:
        return max(1e-10, 1.0 - lam * mu * rho)
    if x == 1 and y == 0:
        return max(1e-10, 1.0 + mu * rho)
    if x == 0 and y == 1:
        return max(1e-10, 1.0 + lam * rho)
    if x == 1 and y == 1:
        return max(1e-10, 1.0 - rho)
    return 1.0


_L2_LAMBDA = 0.005  # L2 regularization: shrinks params toward 0, prevents bound-hitting


def _dc_loglik(
    params: np.ndarray,
    teams: list[str],
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
) -> float:
    """Negative log-likelihood for Dixon-Coles model — fully vectorised + L2 regularization."""
    attack = np.concatenate([[0.0], params[:n_teams - 1]])
    defense = params[n_teams - 1: 2 * n_teams - 1]
    home_adv = params[2 * n_teams - 1]
    rho = params[2 * n_teams]

    lam = np.exp(attack[home_idx] - defense[away_idx] + home_adv).clip(0.01)
    mu = np.exp(attack[away_idx] - defense[home_idx]).clip(0.01)

    # Vectorised tau correction (only modifies scores 0-0, 1-0, 0-1, 1-1)
    x, y = home_goals, away_goals
    tau = np.ones(len(x))
    m00 = (x == 0) & (y == 0)
    m10 = (x == 1) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m11 = (x == 1) & (y == 1)
    tau[m00] = np.maximum(1e-10, 1.0 - lam[m00] * mu[m00] * rho)
    tau[m10] = np.maximum(1e-10, 1.0 + mu[m10] * rho)
    tau[m01] = np.maximum(1e-10, 1.0 + lam[m01] * rho)
    tau[m11] = np.maximum(1e-10, 1.0 - rho)

    # Poisson log-PMF: k*log(lam) - lam - lgamma(k+1)
    from scipy.special import gammaln
    ll_home = x * np.log(lam) - lam - gammaln(x + 1)
    ll_away = y * np.log(mu) - mu - gammaln(y + 1)

    ll = weights @ (np.log(tau) + ll_home + ll_away)
    # L2 penalty on attack+defense (excludes anchored first team and scalar params)
    l2 = _L2_LAMBDA * (np.sum(params[:n_teams - 1] ** 2) + np.sum(defense ** 2))
    return -float(ll) + l2


TOURNAMENT_CUTOFF = pd.Timestamp("2026-06-11")  # WC 2026 kick-off — never train on these


def fit_dixon_coles(
    results_df: pd.DataFrame,
    xi: float = 0.003,
    start_year: int = 2010,
    min_importance: float = 1.5,
) -> dict:
    """
    Fit Dixon-Coles Poisson model with time-decay weighting.

    Returns dict with attack, defense, home_advantage, rho params.
    """
    df = results_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["importance"] = df["tournament"].apply(_importance_weight)

    # Filter — strict cutoff prevents training on WC 2026 results (data leakage)
    cutoff = pd.Timestamp(f"{start_year}-01-01")
    df = df[
        (df["date"] >= cutoff)
        & (df["date"] < TOURNAMENT_CUTOFF)
        & (df["importance"] >= min_importance)
        & df["home_score"].notna()
        & df["away_score"].notna()
    ].copy()

    n_leaked = (results_df["date"] >= TOURNAMENT_CUTOFF).sum() if "date" in results_df.columns else 0
    if n_leaked > 0:
        log.info("Excluded %d WC-2026 rows (date >= %s) to prevent data leakage", n_leaked, TOURNAMENT_CUTOFF.date())

    log.info("Dixon-Coles: training on %d matches", len(df))

    df["days_ago"] = (TODAY - df["date"]).dt.days
    df["time_weight"] = np.exp(-xi * df["days_ago"]) * df["importance"]

    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    team_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    home_idx = df["home_team"].map(team_idx).values.astype(int)
    away_idx = df["away_team"].map(team_idx).values.astype(int)
    home_goals = df["home_score"].values.astype(int)
    away_goals = df["away_score"].values.astype(int)
    weights = df["time_weight"].values

    # Initial params: attack_1..N-1, defense_0..N-1, home_adv, rho
    n_params = (n_teams - 1) + n_teams + 2
    x0 = np.zeros(n_params)
    x0[2 * n_teams - 1] = 0.1   # home_adv
    x0[2 * n_teams] = -0.05     # rho

    # Bounds
    bounds = (
        [(-3.0, 3.0)] * (n_teams - 1)   # attack
        + [(-3.0, 3.0)] * n_teams        # defense
        + [(0.0, 1.0)]                   # home_adv
        + [(-0.5, 0.5)]                  # rho
    )

    log.info("Optimising %d parameters for %d teams ...", n_params, n_teams)
    # maxfun budget: L-BFGS-B uses numerical gradients (~n_params evals/iter)
    # so 1000 full iterations ≈ n_params * 1000 function evaluations
    result = minimize(
        _dc_loglik,
        x0,
        args=(teams, home_idx, away_idx, home_goals, away_goals, weights, n_teams),
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": 2000,
            "maxfun": n_params * 1200,  # ~1200 full quasi-Newton iterations
            "ftol": 1e-8,
            "gtol": 1e-5,
        },
    )

    if not result.success:
        log.warning("Optimisation: %s (nit=%d, nfev=%d)", result.message, result.get("nit", -1), result.get("nfev", -1))

    params = result.x
    attack_vals = np.concatenate([[0.0], params[:n_teams - 1]])
    defense_vals = params[n_teams - 1: 2 * n_teams - 1]
    home_adv = float(params[2 * n_teams - 1])
    rho = float(params[2 * n_teams])

    return {
        "teams": teams,
        "attack": {t: float(attack_vals[i]) for i, t in enumerate(teams)},
        "defense": {t: float(defense_vals[i]) for i, t in enumerate(teams)},
        "home_advantage": home_adv,
        "rho": rho,
        "trained_at": datetime.utcnow().isoformat(),
    }


def dc_match_probs(
    dc_params: dict,
    team_a: str,
    team_b: str,
    is_neutral: bool = True,
    max_goals: int = 8,
) -> dict[str, float]:
    """
    Compute home_win / draw / away_win probabilities from Dixon-Coles parameters.

    team_a is treated as "home" (for the attack/defense calc), is_neutral=True
    removes the home advantage term.
    """
    attack = dc_params["attack"]
    defense = dc_params["defense"]
    home_adv = 0.0 if is_neutral else dc_params["home_advantage"]
    rho = dc_params["rho"]

    # Use global mean if team not in training data
    mean_attack = float(np.mean(list(attack.values())))
    mean_defense = float(np.mean(list(defense.values())))

    a_att = attack.get(team_a, mean_attack)
    a_def = defense.get(team_a, mean_defense)
    b_att = attack.get(team_b, mean_attack)
    b_def = defense.get(team_b, mean_defense)

    lam = max(0.01, np.exp(a_att - b_def + home_adv))
    mu = max(0.01, np.exp(b_att - a_def))

    p_home = p_draw = p_away = 0.0
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            tau = _dc_tau(x, y, lam, mu, rho)
            p = tau * poisson.pmf(x, lam) * poisson.pmf(y, mu)
            if x > y:
                p_home += p
            elif x == y:
                p_draw += p
            else:
                p_away += p

    total = p_home + p_draw + p_away
    if total < 1e-9:
        return {"home_win": 1 / 3, "draw": 1 / 3, "away_win": 1 / 3}

    return {
        "home_win": p_home / total,
        "draw": p_draw / total,
        "away_win": p_away / total,
    }


# ─────────────────────────────────────────────────────────────
# Model B: XGBoost
# ─────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "elo_diff", "elo_home", "elo_away",
    "fifa_pts_diff", "fifa_pts_home", "fifa_pts_away",
    "form_win_home", "form_draw_home", "form_loss_home",
    "form_avg_gf_home", "form_avg_ga_home", "form_pts_home",
    "form_win_away", "form_draw_away", "form_loss_away",
    "form_avg_gf_away", "form_avg_ga_away", "form_pts_away",
    "same_confederation", "is_neutral", "importance",
    "days_since_last_home", "days_since_last_away",
    "h2h_home_wins", "h2h_draws", "h2h_away_wins",
]


def _label_encode_confederation(df: pd.DataFrame) -> pd.DataFrame:
    conf_map = {"UEFA": 0, "CONMEBOL": 1, "CONCACAF": 2, "AFC": 3, "CAF": 4, "OFC": 5, "OTHER": 6}
    df = df.copy()
    df["conf_home_enc"] = df.get("confederation_home", pd.Series("OTHER", index=df.index)).map(conf_map).fillna(6)
    df["conf_away_enc"] = df.get("confederation_away", pd.Series("OTHER", index=df.index)).map(conf_map).fillna(6)
    return df


def train_xgboost(
    features_df: pd.DataFrame,
    n_trials: int = 50,
) -> tuple:
    """
    Train XGBoost 3-way classifier. Returns (model, feature_importance_df).
    """
    try:
        import xgboost as xgb
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as e:
        log.error("Missing dependency: %s", e)
        raise

    df = _label_encode_confederation(features_df)
    df["date"] = pd.to_datetime(df["date"])

    feat_cols = FEATURE_COLS + ["conf_home_enc", "conf_away_enc"]
    # Only keep cols that actually exist
    feat_cols = [c for c in feat_cols if c in df.columns]

    # Train on 2000-2021 competitive matches; validate on 2022 WC
    train_mask = (df["date"].dt.year <= 2021) & (df["importance"] >= 1.0)
    val_mask = (df["date"].dt.year == 2022) & df["tournament"].str.contains("World Cup", case=False, na=False)

    train = df[train_mask].dropna(subset=feat_cols + ["target"])
    val = df[val_mask].dropna(subset=feat_cols + ["target"])

    X_train = train[feat_cols].values.astype(float)
    y_train = train["target"].values.astype(int)
    X_val = val[feat_cols].values.astype(float) if len(val) > 0 else None
    y_val = val["target"].values.astype(int) if len(val) > 0 else None

    log.info("XGBoost: train=%d  val=%d  features=%d", len(X_train), len(val), len(feat_cols))

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "use_label_encoder": False,
            "eval_metric": "mlogloss",
            "objective": "multi:softprob",
            "num_class": 3,
            "verbosity": 0,
        }
        m = xgb.XGBClassifier(**params)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)] if X_val is not None else None,
              verbose=False)
        from sklearn.metrics import log_loss
        preds = m.predict_proba(X_val if X_val is not None else X_train)
        return log_loss(y_val if y_val is not None else y_train, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    log.info("Best XGB params: %s  (val log-loss=%.4f)", best, study.best_value)

    model = xgb.XGBClassifier(
        **best,
        use_label_encoder=False,
        eval_metric="mlogloss",
        objective="multi:softprob",
        num_class=3,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    imp_df = pd.DataFrame({
        "feature": feat_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return model, imp_df, feat_cols


def xgb_match_probs(model, feat_cols: list[str], feature_row: dict) -> dict[str, float]:
    """Compute match probabilities from XGBoost model given a feature dict."""
    row = np.array([feature_row.get(c, 0.0) for c in feat_cols], dtype=float).reshape(1, -1)
    probs = model.predict_proba(row)[0]  # [away_win, draw, home_win] for classes 0,1,2
    return {"home_win": float(probs[2]), "draw": float(probs[1]), "away_win": float(probs[0])}


# ─────────────────────────────────────────────────────────────
# Model C: Elo baseline
# ─────────────────────────────────────────────────────────────

def elo_match_probs(elo_ratings: dict[str, float], team_a: str, team_b: str) -> dict[str, float]:
    """
    Pure Elo win probability; draw probability from a logistic squeeze.
    """
    r_a = elo_ratings.get(team_a, 1500.0)
    r_b = elo_ratings.get(team_b, 1500.0)
    p_a_wins = 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))
    # Draw peaks when teams are evenly matched
    draw_base = 0.28
    draw_prob = draw_base * (1.0 - abs(2.0 * p_a_wins - 1.0))
    home_win = p_a_wins * (1.0 - draw_prob)
    away_win = (1.0 - p_a_wins) * (1.0 - draw_prob)
    return {"home_win": home_win, "draw": draw_prob, "away_win": away_win}


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-xgb", action="store_true", help="Skip XGBoost (slow)")
    parser.add_argument("--xgb-trials", type=int, default=50)
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results_path = RAW_DIR / "international_results.csv"
    if not results_path.exists():
        log.error("Missing %s", results_path)
        sys.exit(1)

    log.info("Loading international results ...")
    results_df = pd.read_csv(results_path)
    results_df.columns = results_df.columns.str.strip()
    if "home_score" not in results_df.columns and "home_goals" in results_df.columns:
        results_df = results_df.rename(columns={"home_goals": "home_score", "away_goals": "away_score"})
    results_df["date"] = pd.to_datetime(results_df["date"])

    # Hard cutoff: never train on WC 2026 results (data leakage guard)
    pre_wc = results_df[results_df["date"] < TOURNAMENT_CUTOFF]
    n_removed = len(results_df) - len(pre_wc)
    if n_removed > 0:
        log.info("Removed %d rows on/after %s (WC 2026 — data leakage guard)", n_removed, TOURNAMENT_CUTOFF.date())
    results_df = pre_wc

    # ── Model C: Elo (fastest, compute first) ──────────────────────
    log.info("Computing Elo ratings ...")
    sys.path.insert(0, str(Path(__file__).parent))
    from feature_engineering import compute_elo_ratings
    elo_ratings = compute_elo_ratings(results_df, cutoff=TOURNAMENT_CUTOFF)

    elo_path = MODELS_DIR / "elo_ratings.json"
    with open(elo_path, "w") as f:
        json.dump(elo_ratings, f, indent=2)
    log.info("Saved Elo ratings for %d teams → %s", len(elo_ratings), elo_path)

    # ── Model A: Dixon-Coles ────────────────────────────────────────
    log.info("Fitting Dixon-Coles model ...")
    dc_params = fit_dixon_coles(results_df)
    dc_path = MODELS_DIR / "dixon_coles_params.json"
    with open(dc_path, "w") as f:
        json.dump(dc_params, f, indent=2)
    log.info("Saved Dixon-Coles params → %s", dc_path)

    # Quick sanity check
    probs = dc_match_probs(dc_params, "France", "Brazil")
    log.info("DC probs France vs Brazil: %s", {k: f"{v:.1%}" for k, v in probs.items()})

    # ── Model B: XGBoost ────────────────────────────────────────────
    if args.skip_xgb:
        log.info("Skipping XGBoost (--skip-xgb)")
        return

    features_path = PROCESSED_DIR / "features.parquet"
    if not features_path.exists():
        log.error("Missing %s — run feature_engineering.py first", features_path)
        sys.exit(1)

    log.info("Loading feature matrix ...")
    features_df = pd.read_parquet(features_path)

    log.info("Training XGBoost (%d optuna trials) ...", args.xgb_trials)
    xgb_model, imp_df, feat_cols = train_xgboost(features_df, n_trials=args.xgb_trials)

    xgb_path = MODELS_DIR / "xgb_wc2026.json"
    xgb_model.save_model(str(xgb_path))

    # Save feature list alongside model
    feat_meta = {"feature_cols": feat_cols}
    with open(MODELS_DIR / "xgb_feature_cols.json", "w") as f:
        json.dump(feat_meta, f, indent=2)

    log.info("Saved XGBoost model → %s", xgb_path)
    log.info("Top 10 features:\n%s", imp_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
