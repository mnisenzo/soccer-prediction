"""
Generate and freeze pre-tournament predictions for all 72 WC 2026 group-stage matches.

IMPORTANT: Once generated, pretournament_predictions.csv is IMMUTABLE.
           This script refuses to overwrite it — re-runs just reload the file.

Usage:
    python backtest/freeze_pretournament.py

Output:
    backtest/pretournament_predictions.csv
"""
from __future__ import annotations

import json
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

OUT_FILE = BACKTEST_DIR / "pretournament_predictions.csv"
FREEZE_MARKER = BACKTEST_DIR / ".predictions_frozen"


# ─── DC helpers ───────────────────────────────────────────────────────────────

def _dc_probs_and_xg(dc: dict, team_a: str, team_b: str) -> dict:
    """
    Compute Dixon-Coles win/draw/loss + xG + modal score + full scoreline matrix.
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

    return {
        "dc_win_a":        p_win,
        "dc_draw":         p_draw,
        "dc_win_b":        p_loss,
        "dc_xg_a":         lam,
        "dc_xg_b":         mu,
        "dc_pred_score_a": pred_score_a,
        "dc_pred_score_b": pred_score_b,
        "dc_scoreline_probs": json.dumps([[round(float(joint[i, j]), 6) for j in range(8)] for i in range(8)]),
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


# ─── Ensemble ─────────────────────────────────────────────────────────────────

def _ensemble(dc_p: dict, elo_p: dict, xgb_p: dict | None) -> dict:
    """
    Weighted ensemble: DC=0.40, XGB=0.40, Elo=0.20.
    If XGB unavailable, re-weight to DC=0.667, Elo=0.333.
    """
    if xgb_p:
        weights = {"dc": 0.40, "xgb": 0.40, "elo": 0.20}
        ens_a = weights["dc"] * dc_p["dc_win_a"] + weights["xgb"] * xgb_p["xgb_win_a"] + weights["elo"] * elo_p["elo_win_a"]
        ens_d = weights["dc"] * dc_p["dc_draw"]  + weights["xgb"] * xgb_p["xgb_draw"]  + weights["elo"] * elo_p["elo_draw"]
        ens_b = weights["dc"] * dc_p["dc_win_b"] + weights["xgb"] * xgb_p["xgb_win_b"] + weights["elo"] * elo_p["elo_win_b"]
    else:
        w_dc, w_elo = 2 / 3, 1 / 3
        ens_a = w_dc * dc_p["dc_win_a"] + w_elo * elo_p["elo_win_a"]
        ens_d = w_dc * dc_p["dc_draw"]  + w_elo * elo_p["elo_draw"]
        ens_b = w_dc * dc_p["dc_win_b"] + w_elo * elo_p["elo_win_b"]

    total = ens_a + ens_d + ens_b
    return {
        "ens_win_a": ens_a / total,
        "ens_draw":  ens_d / total,
        "ens_win_b": ens_b / total,
    }


# ─── XGBoost (optional) ───────────────────────────────────────────────────────

def _load_xgb():
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


def _xgb_probs(model, feat_cols: list, elo: dict, team_a: str, team_b: str) -> dict | None:
    if model is None:
        return None
    ta, tb = to_training_name(team_a), to_training_name(team_b)
    r_a = elo.get(ta, 1500.0)
    r_b = elo.get(tb, 1500.0)
    row = {
        "elo_home": r_a, "elo_away": r_b, "elo_diff": r_a - r_b,
        "fifa_pts_home": 1000.0, "fifa_pts_away": 1000.0, "fifa_pts_diff": 0.0,
        "form_win_home": 0.5, "form_draw_home": 0.2, "form_loss_home": 0.3,
        "form_avg_gf_home": 1.5, "form_avg_ga_home": 1.2, "form_pts_home": 1.7,
        "form_win_away": 0.5, "form_draw_away": 0.2, "form_loss_away": 0.3,
        "form_avg_gf_away": 1.5, "form_avg_ga_away": 1.2, "form_pts_away": 1.7,
        "same_confederation": 0, "is_neutral": 1, "importance": 4.0,
        "days_since_last_home": 7, "days_since_last_away": 7,
        "h2h_home_wins": 0, "h2h_draws": 0, "h2h_away_wins": 0,
        "conf_home_enc": 0, "conf_away_enc": 0,
    }
    x = np.array([row.get(c, 0.0) for c in feat_cols], dtype=float).reshape(1, -1)
    p = model.predict_proba(x)[0]
    return {"xgb_win_a": float(p[2]), "xgb_draw": float(p[1]), "xgb_win_b": float(p[0])}


# ─── Main ─────────────────────────────────────────────────────────────────────

def generate_pretournament_predictions() -> pd.DataFrame:
    """
    Generate predictions for all 72 WC 2026 group-stage matches.
    IMMUTABLE: refuses to overwrite if file already exists.
    """
    BACKTEST_DIR.mkdir(exist_ok=True)

    if OUT_FILE.exists():
        print(f"Predictions already frozen at {OUT_FILE}. Not overwriting.")
        print("Delete the file manually if you genuinely need to regenerate.")
        return pd.read_csv(OUT_FILE)

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

    xgb_model, feat_cols = _load_xgb()
    xgb_available = xgb_model is not None
    print(f"Models loaded: DC + Elo" + (" + XGBoost" if xgb_available else " (XGBoost not found)"))
    print(f"Ensemble weights: DC={'40%' if xgb_available else '66.7%'}  "
          f"XGB={'40%' if xgb_available else 'N/A'}  "
          f"Elo={'20%' if xgb_available else '33.3%'}")

    fixtures = get_all_group_fixtures()
    print(f"Generating predictions for {len(fixtures)} fixtures ...")

    rows = []
    for fix in fixtures:
        g = fix["group"]
        ta = fix["team_a"]
        tb = fix["team_b"]

        dc_p = _dc_probs_and_xg(dc, ta, tb)
        elo_p = _elo_probs(elo, ta, tb)
        xgb_p = _xgb_probs(xgb_model, feat_cols, elo, ta, tb) if xgb_available else None
        ens_p = _ensemble(dc_p, elo_p, xgb_p)

        row: dict = {
            "match_id": fix["match_id"],
            "date": fix["date"],
            "group": g,
            "team_a": ta,
            "team_b": tb,
            **dc_p,
            **(xgb_p if xgb_p else {"xgb_win_a": None, "xgb_draw": None, "xgb_win_b": None}),
            **elo_p,
            **ens_p,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    col_order = [
        "match_id", "date", "group", "team_a", "team_b",
        "dc_win_a", "dc_draw", "dc_win_b", "dc_xg_a", "dc_xg_b",
        "dc_pred_score_a", "dc_pred_score_b", "dc_scoreline_probs",
        "xgb_win_a", "xgb_draw", "xgb_win_b",
        "elo_win_a", "elo_draw", "elo_win_b",
        "ens_win_a", "ens_draw", "ens_win_b",
    ]
    df = df[col_order]
    df.to_csv(OUT_FILE, index=False)

    # Write freeze marker
    FREEZE_MARKER.write_text(
        f"Frozen predictions generated. {len(df)} fixtures. "
        f"XGBoost={'included' if xgb_available else 'omitted'}."
    )

    print(f"Saved frozen predictions to {OUT_FILE}")
    print(f"Sample (Group A):\n{df[df['group']=='A'][['team_a','team_b','dc_win_a','dc_draw','dc_win_b','ens_win_a','ens_win_b']].to_string(index=False)}")
    return df


if __name__ == "__main__":
    generate_pretournament_predictions()
