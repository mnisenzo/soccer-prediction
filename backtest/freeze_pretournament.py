"""
Generate and freeze pre-tournament predictions for all 72 WC 2026 group-stage matches.

IMPORTANT: Once generated, pretournament_predictions.csv is IMMUTABLE.
           This script refuses to overwrite it unless --force is passed.

Usage:
    python backtest/freeze_pretournament.py           # generate once (immutable)
    python backtest/freeze_pretournament.py --force   # regenerate (wipes old freeze)
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "backtest"))

from constants import (
    BACKTEST_DIR, MODELS_DIR, WC2026_GROUPS,
    get_all_group_fixtures, to_training_name,
)
from markets import score_matrix_markets

OUT_FILE = BACKTEST_DIR / "pretournament_predictions.csv"
FREEZE_MARKER = BACKTEST_DIR / ".predictions_frozen"

_TOURNAMENT_CUTOFF = pd.Timestamp("2026-06-11")
_CONF_ENC = {"UEFA": 0, "CONMEBOL": 1, "CONCACAF": 2, "AFC": 3, "CAF": 4, "OFC": 5, "OTHER": 6}

# Binary market columns derived from the DC joint matrix (novel — not duplicating dc_win_a/b/draw)
_MKT_COLS = [
    "dc_over_0_5", "dc_over_1_5", "dc_over_2_5", "dc_over_3_5", "dc_over_4_5",
    "dc_btts", "dc_home_scores", "dc_away_scores",
    "dc_home_cs", "dc_away_cs",
    "dc_home_2plus", "dc_away_2plus",
    "dc_home_3plus", "dc_away_3plus",
]


# ─── DC helpers ───────────────────────────────────────────────────────────────

def _dc_probs_and_xg(dc: dict, team_a: str, team_b: str) -> dict:
    """
    Compute Dixon-Coles win/draw/loss + xG + modal score + full scoreline matrix
    + binary market probabilities derived from the score matrix.
    All on neutral ground (no home_adv).
    """
    from scipy.stats import poisson

    attack = dc["attack"]
    defense = dc["defense"]
    rho = dc["rho"]

    mean_a = float(np.mean(list(attack.values())))
    mean_d = float(np.mean(list(defense.values())))

    ta, tb = to_training_name(team_a), to_training_name(team_b)
    a_att = attack.get(ta, mean_a)
    a_def = defense.get(ta, mean_d)
    b_att = attack.get(tb, mean_a)
    b_def = defense.get(tb, mean_d)

    lam = float(max(0.01, np.exp(a_att - b_def)))  # team_a expected goals
    mu  = float(max(0.01, np.exp(b_att - a_def)))  # team_b expected goals

    # 8×8 scoreline probability matrix (0..7 goals each)
    scores = np.arange(8)
    p_x = poisson.pmf(scores, lam)
    p_y = poisson.pmf(scores, mu)
    joint = np.outer(p_x, p_y)   # shape (8,8): joint[i,j] = P(score_a=i, score_b=j)

    # Dixon-Coles tau correction on low-scoring cells
    joint[0, 0] *= max(1e-10, 1.0 - lam * mu * rho)
    joint[1, 0] *= max(1e-10, 1.0 + mu * rho)
    joint[0, 1] *= max(1e-10, 1.0 + lam * rho)
    joint[1, 1] *= max(1e-10, 1.0 - rho)

    X, Y = np.meshgrid(scores, scores, indexing="ij")
    p_win  = float(joint[X > Y].sum())
    p_draw = float(joint[X == Y].sum())
    p_loss = float(joint[X < Y].sum())
    total  = p_win + p_draw + p_loss

    if total < 1e-9:
        p_win = p_draw = p_loss = 1 / 3
    else:
        p_win, p_draw, p_loss = p_win / total, p_draw / total, p_loss / total

    # Modal (most likely) scoreline
    modal_idx = np.unravel_index(joint.argmax(), joint.shape)
    pred_score_a, pred_score_b = int(modal_idx[0]), int(modal_idx[1])

    # Binary markets from score matrix
    mkt = score_matrix_markets(joint)

    return {
        "dc_win_a":        p_win,
        "dc_draw":         p_draw,
        "dc_win_b":        p_loss,
        "dc_xg_a":         lam,
        "dc_xg_b":         mu,
        "dc_pred_score_a": pred_score_a,
        "dc_pred_score_b": pred_score_b,
        "dc_scoreline_probs": json.dumps([[round(float(joint[i, j]), 6) for j in range(8)] for i in range(8)]),
        # Binary markets (prefixed dc_ = derived from DC joint matrix)
        "dc_over_0_5":    mkt["over_0_5"],
        "dc_over_1_5":    mkt["over_1_5"],
        "dc_over_2_5":    mkt["over_2_5"],
        "dc_over_3_5":    mkt["over_3_5"],
        "dc_over_4_5":    mkt["over_4_5"],
        "dc_btts":        mkt["btts"],
        "dc_home_scores": mkt["home_scores"],
        "dc_away_scores": mkt["away_scores"],
        "dc_home_cs":     mkt["home_cs"],
        "dc_away_cs":     mkt["away_cs"],
        "dc_home_2plus":  mkt["home_2plus"],
        "dc_away_2plus":  mkt["away_2plus"],
        "dc_home_3plus":  mkt["home_3plus"],
        "dc_away_3plus":  mkt["away_3plus"],
    }


# ─── Elo helpers ──────────────────────────────────────────────────────────────

def _elo_probs(elo: dict, team_a: str, team_b: str) -> dict:
    ta, tb = to_training_name(team_a), to_training_name(team_b)
    r_a = elo.get(ta, elo.get(team_a, 1500.0))
    r_b = elo.get(tb, elo.get(team_b, 1500.0))
    p_a = 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))
    draw_base = 0.28
    draw_prob = draw_base * (1.0 - abs(2.0 * p_a - 1.0))
    return {
        "elo_win_a": p_a * (1.0 - draw_prob),
        "elo_draw":  draw_prob,
        "elo_win_b": (1.0 - p_a) * (1.0 - draw_prob),
    }


# ─── XGBoost team feature precomputation ─────────────────────────────────────

def _precompute_team_features(cutoff: pd.Timestamp) -> dict[str, dict]:
    """
    Compute real rolling form features for every WC 2026 team from historical
    results up to cutoff. Replaces the hardcoded population-mean constants.
    """
    results_path = PROJECT_ROOT / "data" / "raw" / "international_results.csv"
    if not results_path.exists():
        print(f"  Warning: {results_path} not found — XGB will use fallback constants")
        return {}

    try:
        from feature_engineering import compute_rolling_form, _last_match_date, CONFEDERATION_MAP
    except ImportError as e:
        print(f"  Warning: cannot import feature_engineering for XGB features: {e}")
        return {}

    df = pd.read_csv(results_path)
    df.columns = df.columns.str.strip()
    if "home_score" not in df.columns and "home_goals" in df.columns:
        df = df.rename(columns={"home_goals": "home_score", "away_goals": "away_score"})
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] < cutoff].copy()

    all_teams = {t for teams in WC2026_GROUPS.values() for t in teams}
    features: dict[str, dict] = {}
    for team in all_teams:
        form = compute_rolling_form(df, team, cutoff, n_matches=10)
        last = _last_match_date(df, team, cutoff)
        days_since = int((cutoff - last).days) if last else 90
        conf = CONFEDERATION_MAP.get(team, "OTHER")
        features[team] = {
            "win_rate":        form["win_rate"],
            "draw_rate":       form["draw_rate"],
            "loss_rate":       form["loss_rate"],
            "avg_gf":          form["avg_gf"],
            "avg_ga":          form["avg_ga"],
            "pts_per_game":    form["pts_per_game"],
            "days_since_last": days_since,
            "confederation":   conf,
            "conf_enc":        _CONF_ENC.get(conf, 6),
        }

    print(f"  XGB features precomputed for {len(features)} teams")
    return features


# ─── Ensemble ─────────────────────────────────────────────────────────────────

def _load_ensemble_weights(has_xgb: bool, has_lgbm: bool) -> dict:
    ew_path = MODELS_DIR / "ensemble_weights.json"
    if ew_path.exists():
        with open(ew_path) as f:
            ew = json.load(f)
        print(f"  Ensemble weights from file: DC={ew.get('dc',0):.2f} XGB={ew.get('xgb',0):.2f} "
              f"LGBM={ew.get('lgbm',0):.2f} Elo={ew.get('elo',0):.2f}")
        return ew
    # Default weights
    if has_xgb and has_lgbm:
        return {"dc": 0.35, "xgb": 0.30, "lgbm": 0.15, "elo": 0.20}
    elif has_xgb:
        return {"dc": 0.40, "xgb": 0.40, "elo": 0.20}
    else:
        return {"dc": 0.667, "elo": 0.333}


def _ensemble(dc_p: dict, elo_p: dict, xgb_p: dict | None, lgbm_p: dict | None,
              weights: dict) -> dict:
    """Weighted ensemble using provided weights."""
    ens_a = ens_d = ens_b = 0.0
    total_w = 0.0

    ens_a += weights["dc"] * dc_p["dc_win_a"]
    ens_d += weights["dc"] * dc_p["dc_draw"]
    ens_b += weights["dc"] * dc_p["dc_win_b"]
    total_w += weights["dc"]

    ens_a += weights["elo"] * elo_p["elo_win_a"]
    ens_d += weights["elo"] * elo_p["elo_draw"]
    ens_b += weights["elo"] * elo_p["elo_win_b"]
    total_w += weights["elo"]

    if xgb_p and weights.get("xgb", 0) > 0:
        ens_a += weights["xgb"] * xgb_p["xgb_win_a"]
        ens_d += weights["xgb"] * xgb_p["xgb_draw"]
        ens_b += weights["xgb"] * xgb_p["xgb_win_b"]
        total_w += weights["xgb"]

    if lgbm_p and weights.get("lgbm", 0) > 0:
        ens_a += weights["lgbm"] * lgbm_p["lgbm_win_a"]
        ens_d += weights["lgbm"] * lgbm_p["lgbm_draw"]
        ens_b += weights["lgbm"] * lgbm_p["lgbm_win_b"]
        total_w += weights["lgbm"]

    if total_w < 1e-9:
        return {"ens_win_a": 1/3, "ens_draw": 1/3, "ens_win_b": 1/3}
    total_p = ens_a + ens_d + ens_b
    if total_p < 1e-9:
        total_p = 1.0
    return {
        "ens_win_a": ens_a / total_p,
        "ens_draw":  ens_d / total_p,
        "ens_win_b": ens_b / total_p,
    }


# ─── XGBoost (optional) ───────────────────────────────────────────────────────

def _load_xgb():
    # Prefer calibrated model if available
    cal_path = MODELS_DIR / "xgb_calibrated.pkl"
    if cal_path.exists():
        try:
            with open(cal_path, "rb") as f:
                d = pickle.load(f)
            print("  XGBoost: loaded calibrated model")
            return d["model"], d["feature_cols"]
        except Exception as e:
            print(f"  XGBoost calibrated load failed: {e}, trying uncalibrated ...")

    xgb_path = MODELS_DIR / "xgb_wc2026.json"
    feat_path = MODELS_DIR / "xgb_feature_cols.json"
    if not xgb_path.exists() or not feat_path.exists():
        return None, None
    try:
        import xgboost as xgb
        m = xgb.XGBClassifier()
        m.load_model(str(xgb_path))
        with open(feat_path) as f:
            feat_cols = json.load(f)["feature_cols"]
        return m, feat_cols
    except Exception as e:
        print(f"  XGBoost unavailable: {e}")
        return None, None


def _load_lgbm():
    for path in [MODELS_DIR / "lgbm_calibrated.pkl", MODELS_DIR / "lgbm_wc2026.pkl"]:
        if path.exists():
            try:
                with open(path, "rb") as f:
                    d = pickle.load(f)
                print(f"  LightGBM: loaded {path.name}")
                return d["model"], d["feature_cols"]
            except Exception as e:
                print(f"  LightGBM load failed from {path.name}: {e}")
    return None, None


def _build_feature_row(elo: dict, team_features: dict, team_a: str, team_b: str,
                        feat_cols: list) -> np.ndarray:
    """Build a model input row from team names, Elo, and precomputed form features."""
    ta, tb = to_training_name(team_a), to_training_name(team_b)
    r_a = elo.get(ta, elo.get(team_a, 1500.0))
    r_b = elo.get(tb, elo.get(team_b, 1500.0))

    fa = team_features.get(team_a, team_features.get(ta, {}))
    fb = team_features.get(team_b, team_features.get(tb, {}))
    same_conf = int(fa.get("confederation", "A") == fb.get("confederation", "B"))

    row = {
        # Elo (new names)
        "elo_diff": r_a - r_b, "elo_h": r_a, "elo_a": r_b,
        # Home rolling form (new names)
        "h_r10_win":   fa.get("win_rate",     0.4),
        "h_r10_draw":  fa.get("draw_rate",    0.25),
        "h_r10_loss":  fa.get("loss_rate",    0.35),
        "h_r10_gf":    fa.get("avg_gf",       1.3),
        "h_r10_ga":    fa.get("avg_ga",       1.3),
        "h_r10_gd":    fa.get("avg_gf", 1.3) - fa.get("avg_ga", 1.3),
        "h_r10_pts":   fa.get("pts_per_game", 1.45),
        "h_days_rest": fa.get("days_since_last", 14),
        # Away rolling form (new names)
        "a_r10_win":   fb.get("win_rate",     0.4),
        "a_r10_draw":  fb.get("draw_rate",    0.25),
        "a_r10_loss":  fb.get("loss_rate",    0.35),
        "a_r10_gf":    fb.get("avg_gf",       1.3),
        "a_r10_ga":    fb.get("avg_ga",       1.3),
        "a_r10_gd":    fb.get("avg_gf", 1.3) - fb.get("avg_ga", 1.3),
        "a_r10_pts":   fb.get("pts_per_game", 1.45),
        "a_days_rest": fb.get("days_since_last", 14),
        # H2H / WC history
        "h2h_home_wins": 0, "h2h_draws": 0, "h2h_away_wins": 0,
        "wc_apps_home": fa.get("wc_apps", 5), "wc_wins_home": fa.get("wc_wins", 3),
        "wc_apps_away": fb.get("wc_apps", 5), "wc_wins_away": fb.get("wc_wins", 3),
        # Confederation
        "conf_home_enc": fa.get("conf_enc", 6), "conf_away_enc": fb.get("conf_enc", 6),
        "same_confederation": same_conf,
        "is_neutral": 1, "importance": 4.0,
    }
    arr = np.array([row.get(c, 0.0) for c in feat_cols], dtype=float).reshape(1, -1)
    return pd.DataFrame(arr, columns=feat_cols)


def _xgb_probs(model, feat_cols: list, elo: dict, team_features: dict,
               team_a: str, team_b: str) -> dict | None:
    if model is None:
        return None
    x = _build_feature_row(elo, team_features, team_a, team_b, feat_cols)
    p = model.predict_proba(x)[0]
    return {"xgb_win_a": float(p[2]), "xgb_draw": float(p[1]), "xgb_win_b": float(p[0])}


def _lgbm_probs(model, feat_cols: list, elo: dict, team_features: dict,
                team_a: str, team_b: str) -> dict | None:
    if model is None:
        return None
    x = _build_feature_row(elo, team_features, team_a, team_b, feat_cols)
    p = model.predict_proba(x)[0]
    return {"lgbm_win_a": float(p[2]), "lgbm_draw": float(p[1]), "lgbm_win_b": float(p[0])}


# ─── Main ─────────────────────────────────────────────────────────────────────

def generate_pretournament_predictions(force: bool = False) -> pd.DataFrame:
    """
    Generate predictions for all 72 WC 2026 group-stage matches.
    Pass force=True to wipe and regenerate an existing freeze.
    """
    BACKTEST_DIR.mkdir(exist_ok=True)

    if OUT_FILE.exists() and not force:
        print(f"Predictions already frozen at {OUT_FILE}.")
        print("Pass --force to regenerate (wipes existing freeze).")
        return pd.read_csv(OUT_FILE)

    if OUT_FILE.exists() and force:
        OUT_FILE.unlink()
        if FREEZE_MARKER.exists():
            FREEZE_MARKER.unlink()
        print("Wiped existing freeze — regenerating ...")

    # Load models
    dc_path = MODELS_DIR / "dixon_coles_params.json"
    elo_path = MODELS_DIR / "elo_ratings.json"

    if not dc_path.exists() or not elo_path.exists():
        raise FileNotFoundError(
            f"Models not found in {MODELS_DIR}. Run src/train_models.py first."
        )

    with open(dc_path) as f:
        dc = json.load(f)
    with open(elo_path) as f:
        elo = json.load(f)

    xgb_model, xgb_feat_cols = _load_xgb()
    lgbm_model, lgbm_feat_cols = _load_lgbm()
    xgb_available = xgb_model is not None
    lgbm_available = lgbm_model is not None
    print(f"Models loaded: DC + Elo"
          + (" + XGBoost" if xgb_available else "")
          + (" + LightGBM" if lgbm_available else ""))

    team_features = _precompute_team_features(_TOURNAMENT_CUTOFF)
    ew = _load_ensemble_weights(xgb_available, lgbm_available)

    fixtures = get_all_group_fixtures()
    print(f"Generating predictions for {len(fixtures)} fixtures ...")

    rows = []
    for fix in fixtures:
        g = fix["group"]
        ta = fix["team_a"]
        tb = fix["team_b"]

        dc_p   = _dc_probs_and_xg(dc, ta, tb)
        elo_p  = _elo_probs(elo, ta, tb)
        xgb_p  = _xgb_probs(xgb_model, xgb_feat_cols, elo, team_features, ta, tb) if xgb_available else None
        lgbm_p = _lgbm_probs(lgbm_model, lgbm_feat_cols, elo, team_features, ta, tb) if lgbm_available else None
        ens_p  = _ensemble(dc_p, elo_p, xgb_p, lgbm_p, ew)

        row: dict = {
            "match_id": fix["match_id"],
            "date": fix["date"],
            "group": g,
            "team_a": ta,
            "team_b": tb,
            **dc_p,
            **(xgb_p if xgb_p else {"xgb_win_a": None, "xgb_draw": None, "xgb_win_b": None}),
            **(lgbm_p if lgbm_p else {"lgbm_win_a": None, "lgbm_draw": None, "lgbm_win_b": None}),
            **elo_p,
            **ens_p,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    col_order = [
        "match_id", "date", "group", "team_a", "team_b",
        "dc_win_a", "dc_draw", "dc_win_b", "dc_xg_a", "dc_xg_b",
        "dc_pred_score_a", "dc_pred_score_b", "dc_scoreline_probs",
        *_MKT_COLS,
        "xgb_win_a", "xgb_draw", "xgb_win_b",
        "lgbm_win_a", "lgbm_draw", "lgbm_win_b",
        "elo_win_a", "elo_draw", "elo_win_b",
        "ens_win_a", "ens_draw", "ens_win_b",
    ]
    # Only keep columns that actually exist
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]
    df.to_csv(OUT_FILE, index=False)

    FREEZE_MARKER.write_text(
        f"Frozen predictions generated. {len(df)} fixtures. "
        f"XGBoost={'included' if xgb_available else 'omitted'}. "
        f"LightGBM={'included' if lgbm_available else 'omitted'}. "
        f"Markets={len(_MKT_COLS)} binary columns."
    )

    print(f"\nSaved frozen predictions to {OUT_FILE}")
    print(f"Sample (Group A):\n{df[df['group']=='A'][['team_a','team_b','dc_win_a','dc_draw','dc_win_b','ens_win_a','ens_win_b']].to_string(index=False)}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Freeze pre-tournament WC 2026 predictions")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if predictions are already frozen",
    )
    args = parser.parse_args()
    generate_pretournament_predictions(force=args.force)
