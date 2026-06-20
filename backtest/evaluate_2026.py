"""
Evaluate WC 2026 pre-tournament predictions against completed matches.

Sections:
  A. Match outcome metrics (RPS, log-loss, accuracy, bootstrap CI)
  B. Goals prediction metrics (MAE, RMSE, bias, scoreline log-P)
  C. Combined summary table

Usage:
    python backtest/evaluate_2026.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson as sp_poisson

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backtest"))

from constants import BACKTEST_DIR

PRED_FILE    = BACKTEST_DIR / "pretournament_predictions.csv"
RESULTS_FILE = BACKTEST_DIR / "completed_results.csv"
METRICS_FILE = BACKTEST_DIR / "metrics_2026.json"


# ─── Outcome metric functions ────────────────────────────────────────────────

def rps(p_win: float, p_draw: float, p_loss: float, outcome: int) -> float:
    """RPS for 3-outcome (2=A win, 1=draw, 0=B win). Lower is better."""
    f1 = float(p_win)
    f2 = float(p_win) + float(p_draw)
    o1 = 1.0 if outcome == 2 else 0.0
    o2 = 0.0 if outcome == 0 else 1.0
    return 0.5 * ((f1 - o1) ** 2 + (f2 - o2) ** 2)


def log_loss_score(p_win: float, p_draw: float, p_loss: float, outcome: int) -> float:
    probs = {2: max(p_win, 1e-9), 1: max(p_draw, 1e-9), 0: max(p_loss, 1e-9)}
    return -np.log(probs[outcome])


def predicted_outcome(p_win: float, p_draw: float, p_loss: float) -> int:
    return max((p_win, 2), (p_draw, 1), (p_loss, 0))[1]


def bootstrap_mean_ci(values: np.ndarray, n: int = 10_000, alpha: float = 0.05) -> tuple:
    rng = np.random.default_rng(42)
    boots = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def expected_calibration_error(
    probs: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, list[dict]]:
    """
    Confidence-based ECE. probs[i] = model confidence, correct[i] = 1 if that prediction was right.
    Returns (ece_scalar, bin_data) where bin_data drives reliability diagrams.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(probs)
    ece = 0.0
    bin_data: list[dict] = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = float(probs[mask].mean())
        bin_acc  = float(correct[mask].mean())
        bin_n    = int(mask.sum())
        ece += (bin_n / n) * abs(bin_acc - bin_conf)
        bin_data.append({
            "bin_mid": round((lo + hi) / 2, 3),
            "confidence": round(bin_conf, 4),
            "accuracy":   round(bin_acc, 4),
            "n":          bin_n,
        })
    return round(ece, 6), bin_data


def per_class_ece(
    p_win: np.ndarray,
    p_draw: np.ndarray,
    p_loss: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Per-class binary ECE treating each 1X2 outcome as a binary calibration problem."""
    ece_w, _ = expected_calibration_error(p_win,  (outcomes == 2).astype(float), n_bins)
    ece_d, _ = expected_calibration_error(p_draw, (outcomes == 1).astype(float), n_bins)
    ece_l, _ = expected_calibration_error(p_loss, (outcomes == 0).astype(float), n_bins)
    return {"win": ece_w, "draw": ece_d, "loss": ece_l, "mean": round((ece_w + ece_d + ece_l) / 3, 6)}


# ─── Model definitions ────────────────────────────────────────────────────────

MODELS = {
    "dc":  ("dc_win_a",  "dc_draw",  "dc_win_b"),
    "elo": ("elo_win_a", "elo_draw", "elo_win_b"),
    "ens": ("ens_win_a", "ens_draw", "ens_win_b"),
}

UNIFORM = (1 / 3, 1 / 3, 1 / 3)


# ─── Section A: match outcome evaluation ─────────────────────────────────────

def evaluate_outcomes(merged: pd.DataFrame) -> dict[str, dict]:
    n = len(merged)
    all_models = dict(MODELS)
    if "xgb_win_a" in merged.columns and merged["xgb_win_a"].notna().any():
        all_models["xgb"] = ("xgb_win_a", "xgb_draw", "xgb_win_b")

    results: dict[str, dict] = {}

    for name, (ca, cd, cb) in all_models.items():
        if ca not in merged.columns:
            continue
        sub = merged[[ca, cd, cb, "outcome"]].dropna()
        if sub.empty:
            continue

        p_w = sub[ca].values.astype(float)
        p_d = sub[cd].values.astype(float)
        p_l = sub[cb].values.astype(float)
        outs = sub["outcome"].values.astype(int)

        rps_vals = np.array([rps(r[ca], r[cd], r[cb], r["outcome"]) for _, r in sub.iterrows()])
        ll_vals  = np.array([log_loss_score(r[ca], r[cd], r[cb], r["outcome"]) for _, r in sub.iterrows()])
        preds    = np.array([predicted_outcome(r[ca], r[cd], r[cb]) for _, r in sub.iterrows()])
        acc      = float(np.mean(preds == outs))
        ci       = bootstrap_mean_ci(rps_vals)

        # ECE: confidence = max predicted prob; correct = argmax matches actual
        max_probs = np.stack([p_w, p_d, p_l], axis=1).max(axis=1)
        correct   = (preds == outs).astype(float)
        ece_conf, cal_bins = expected_calibration_error(max_probs, correct)
        ece_per_class = per_class_ece(p_w, p_d, p_l, outs)

        results[name] = {
            "n": len(sub),
            "rps_mean":    float(rps_vals.mean()),
            "rps_ci_lo":   ci[0],
            "rps_ci_hi":   ci[1],
            "rps_per_match": rps_vals.tolist(),
            "log_loss":    float(ll_vals.mean()),
            "accuracy":    acc,
            "ece_confidence": ece_conf,
            "ece_per_class":  ece_per_class,
            "calibration_bins": cal_bins,
        }

    # Uniform baseline
    rps_u  = np.array([rps(*UNIFORM, o) for o in merged["outcome"]])
    ll_u   = np.array([log_loss_score(*UNIFORM, o) for o in merged["outcome"]])
    rng    = np.random.default_rng(42)
    rand_preds = rng.choice([2, 1, 0], size=n)
    results["baseline_uniform"] = {
        "n": n,
        "rps_mean":    float(rps_u.mean()),
        "rps_ci_lo":   bootstrap_mean_ci(rps_u)[0],
        "rps_ci_hi":   bootstrap_mean_ci(rps_u)[1],
        "rps_per_match": rps_u.tolist(),
        "log_loss":    float(ll_u.mean()),
        "accuracy":    float((merged["outcome"].values == rand_preds).mean()),
    }

    return results


# ─── Section B: goals evaluation ─────────────────────────────────────────────

# Teams considered "strong" — top-15 market odds (American odds <= 6000)
STRONG_TEAMS = {
    "France", "Spain", "England", "Argentina", "Portugal", "Brazil",
    "Germany", "Netherlands", "Norway", "Morocco", "USA", "Belgium",
    "Colombia", "Japan", "Mexico",
}

NAIVE_XG = 1.35  # naive baseline: always predict 1.35 goals per team


def _naive_scoreline_logp(score_a: int, score_b: int) -> float:
    return np.log(sp_poisson.pmf(score_a, NAIVE_XG) * sp_poisson.pmf(score_b, NAIVE_XG) + 1e-12)


def evaluate_goals(merged: pd.DataFrame) -> dict:
    needed = ["dc_xg_a", "dc_xg_b", "dc_pred_score_a", "dc_pred_score_b",
              "score_a", "score_b"]
    if not all(c in merged.columns for c in needed):
        return {}

    sub = merged.dropna(subset=needed).copy()
    if sub.empty:
        return {}

    xa = sub["dc_xg_a"].values.astype(float)
    xb = sub["dc_xg_b"].values.astype(float)
    sa = sub["score_a"].values.astype(float)
    sb = sub["score_b"].values.astype(float)

    mae_a = float(np.abs(xa - sa).mean())
    mae_b = float(np.abs(xb - sb).mean())
    rmse_a = float(np.sqrt(((xa - sa) ** 2).mean()))
    rmse_b = float(np.sqrt(((xb - sb) ** 2).mean()))

    total_mae = float(np.abs((xa + xb) - (sa + sb)).mean())
    gd_mae    = float(np.abs((xa - xb) - (sa - sb)).mean())

    non_draw = (sa != sb)
    if non_draw.sum() > 0:
        dir_correct = float(((np.sign(xa[non_draw] - xb[non_draw]) == np.sign(sa[non_draw] - sb[non_draw]))).mean())
    else:
        dir_correct = float("nan")

    exact = float(((sub["dc_pred_score_a"].values.astype(int) == sa.astype(int)) &
                   (sub["dc_pred_score_b"].values.astype(int) == sb.astype(int))).mean())
    within1 = float(((np.abs(sub["dc_pred_score_a"].values.astype(float) - sa) <= 1) &
                     (np.abs(sub["dc_pred_score_b"].values.astype(float) - sb) <= 1)).mean())

    bias_a = float((xa - sa).mean())
    bias_b = float((xb - sb).mean())

    # Strong vs weak bias
    strong_mask = sub["team_a"].isin(STRONG_TEAMS) | sub["team_b"].isin(STRONG_TEAMS)
    bias_strong = float((xa[strong_mask] - sa[strong_mask]).mean()) if strong_mask.any() else float("nan")
    bias_weak   = float((xa[~strong_mask] - sa[~strong_mask]).mean()) if (~strong_mask).any() else float("nan")

    # Scoreline log-probability
    log_p_vals = []
    naive_log_p_vals = []
    has_matrix = "dc_scoreline_probs" in sub.columns and sub["dc_scoreline_probs"].notna().any()
    for _, row in sub.iterrows():
        si, sj = int(row["score_a"]), int(row["score_b"])
        naive_log_p_vals.append(_naive_scoreline_logp(si, sj))
        if has_matrix and pd.notna(row.get("dc_scoreline_probs")):
            try:
                mat = json.loads(row["dc_scoreline_probs"])
                p = mat[min(si, 7)][min(sj, 7)] if si < 8 and sj < 8 else 1e-9
                log_p_vals.append(np.log(max(float(p), 1e-12)))
            except Exception:
                log_p_vals.append(float("nan"))
        else:
            log_p_vals.append(float("nan"))

    mean_log_p       = float(np.nanmean(log_p_vals)) if log_p_vals else float("nan")
    naive_mean_log_p = float(np.mean(naive_log_p_vals))

    return {
        "n": len(sub),
        "mae_a": mae_a, "mae_b": mae_b, "mae_avg": (mae_a + mae_b) / 2,
        "rmse_a": rmse_a, "rmse_b": rmse_b,
        "total_goals_mae": total_mae,
        "goal_diff_mae": gd_mae,
        "directional_accuracy": dir_correct,
        "exact_score_acc": exact,
        "within1_acc": within1,
        "scoreline_log_p": mean_log_p,
        "naive_log_p": naive_mean_log_p,
        "bias_a": bias_a, "bias_b": bias_b,
        "bias_strong": bias_strong,
        "bias_weak": bias_weak,
    }


# ─── Section B: per-match goals table ────────────────────────────────────────

def print_goals_table(merged: pd.DataFrame) -> str:
    needed = ["dc_xg_a", "dc_xg_b", "dc_pred_score_a", "dc_pred_score_b", "score_a", "score_b"]
    if not all(c in merged.columns for c in needed):
        return ""

    sub = merged.dropna(subset=needed).sort_values("date")
    lines = []
    for _, r in sub.iterrows():
        ta, tb = str(r["team_a"]), str(r["team_b"])
        xa, xb = float(r["dc_xg_a"]), float(r["dc_xg_b"])
        sa, sb = int(r["score_a"]), int(r["score_b"])
        pa, pb = int(r["dc_pred_score_a"]), int(r["dc_pred_score_b"])
        date_s = str(r.get("date", ""))[:10]
        grp    = str(r.get("group", "?"))

        lines.append(f"\n{ta} vs {tb}  ({date_s}, Group {grp})")
        lines.append(f"  Predicted: {ta} {xa:.2f} xG — {tb} {xb:.2f} xG  |  Modal: {pa}-{pb}")
        lines.append(f"  Actual:    {ta} {sa}       — {tb} {sb}")
        lines.append(f"  xG error:  {ta} {xa - sa:+.2f}  |  {tb} {xb - sb:+.2f}")

        if "dc_scoreline_probs" in r.index and pd.notna(r.get("dc_scoreline_probs")):
            try:
                mat = json.loads(r["dc_scoreline_probs"])
                p = mat[min(sa, 7)][min(sb, 7)]
                lines.append(f"  P({sa}-{sb}) = {float(p):.4f}")
            except Exception:
                pass
        lines.append("  " + "-" * 60)

    return "\n".join(lines)


# ─── Merge helper ─────────────────────────────────────────────────────────────

def merge_predictions_results(preds: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    preds   = preds.copy()
    results = results.copy()

    merged_fwd = preds.merge(
        results[["group", "team_a", "team_b", "score_a", "score_b", "outcome", "date"]],
        on=["group", "team_a", "team_b"], how="inner", suffixes=("", "_actual"),
    )

    results_rev = results.copy()
    results_rev["team_a"], results_rev["team_b"] = results["team_b"].copy(), results["team_a"].copy()
    results_rev["score_a"], results_rev["score_b"] = results["score_b"].copy(), results["score_a"].copy()
    results_rev["outcome"] = results_rev["outcome"].map({2: 0, 1: 1, 0: 2})

    merged_rev = preds.merge(
        results_rev[["group", "team_a", "team_b", "score_a", "score_b", "outcome", "date"]],
        on=["group", "team_a", "team_b"], how="inner", suffixes=("", "_actual"),
    )

    all_merged = pd.concat([merged_fwd, merged_rev], ignore_index=True)
    all_merged = all_merged.drop_duplicates(subset=["team_a", "team_b", "group"])

    if "date_actual" in all_merged.columns:
        all_merged["date"] = all_merged["date_actual"].fillna(all_merged["date"])
        all_merged.drop(columns=["date_actual"], inplace=True)

    return all_merged.sort_values("date").reset_index(drop=True)


# ─── Upset detection ──────────────────────────────────────────────────────────

def detect_upsets(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in merged.iterrows():
        pa, pd_, pb = float(r["ens_win_a"]), float(r["ens_draw"]), float(r["ens_win_b"])
        pred    = predicted_outcome(pa, pd_, pb)
        outcome = int(r["outcome"])
        p_act   = {2: pa, 1: pd_, 0: pb}[outcome]
        r_score = rps(pa, pd_, pb, outcome)
        rows.append({
            "team_a": r["team_a"], "team_b": r["team_b"],
            "score": f"{int(r.get('score_a',0))}-{int(r.get('score_b',0))}",
            "outcome_label": {2: "A wins", 1: "Draw", 0: "B wins"}[outcome],
            "predicted_label": {2: "A wins", 1: "Draw", 0: "B wins"}[pred],
            "ens_prob_a": pa, "ens_draw": pd_, "ens_prob_b": pb,
            "prob_of_actual": p_act,
            "rps": r_score,
            "is_upset": (pred != outcome),
            "group": r.get("group", "?"),
            "date": r.get("date", ""),
        })
    return pd.DataFrame(rows).sort_values("rps", ascending=False)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if not PRED_FILE.exists():
        print(f"ERROR: {PRED_FILE} not found.")
        print("Run: python backtest/freeze_pretournament.py first.")
        sys.exit(1)

    if not RESULTS_FILE.exists():
        print(f"ERROR: {RESULTS_FILE} not found.")
        sys.exit(1)

    preds   = pd.read_csv(PRED_FILE)
    results = pd.read_csv(RESULTS_FILE)
    print(f"Predictions: {len(preds)} fixtures  |  Completed: {len(results)} matches")

    merged = merge_predictions_results(preds, results)
    n = len(merged)
    print(f"Matched: {n} matches\n")

    if merged.empty:
        print("No matches merged — check team name consistency.")
        sys.exit(1)

    # ── Section A ──────────────────────────────────────────────────────────────
    outcome_metrics = evaluate_outcomes(merged)
    goals_metrics   = evaluate_goals(merged)

    # ── Section C: summary ─────────────────────────────────────────────────────
    print("=" * 70)
    print(f"=== 2026 WC BACKTEST — {n} COMPLETED MATCHES ===")
    print("=" * 70)
    print()
    print("OUTCOME PREDICTION")
    print(f"{'':22} {'RPS':>7}  {'Log-Loss':>10}  {'Accuracy':>8}  {'ECE':>7}")
    print("-" * 65)
    order  = ["dc", "xgb", "elo", "ens", "baseline_uniform"]
    labels = {"dc": "Dixon-Coles", "xgb": "XGBoost", "elo": "Elo",
               "ens": "Ensemble", "baseline_uniform": "Baseline (uniform)"}
    for k in order:
        if k not in outcome_metrics:
            continue
        m = outcome_metrics[k]
        ece_str = f"{m.get('ece_confidence', float('nan')):.4f}" if "ece_confidence" in m else "  n/a"
        print(f"  {labels[k]:<20} {m['rps_mean']:>7.4f}  {m['log_loss']:>10.4f}  {m['accuracy']:>7.1%}  {ece_str:>7}")

    # Bootstrap CI
    if "ens" in outcome_metrics and "baseline_uniform" in outcome_metrics:
        diff = np.array(outcome_metrics["ens"]["rps_per_match"]) - np.array(outcome_metrics["baseline_uniform"]["rps_per_match"])
        ci_lo, ci_hi = bootstrap_mean_ci(diff)
        print(f"\nBootstrap 95% CI on Ensemble vs Uniform RPS: [{ci_lo:.4f}, {ci_hi:.4f}]")
        if ci_lo <= 0 <= ci_hi:
            print(f"WARNING: difference not yet statistically significant at n={n} matches.")

    # Goals section
    if goals_metrics:
        g = goals_metrics
        print()
        print("GOALS PREDICTION (Dixon-Coles)")
        print(f"{'':30} {'Team A':>8}  {'Team B':>8}  {'Avg':>8}")
        print("-" * 60)
        print(f"  {'MAE (xG vs actual)':<28} {g['mae_a']:>8.3f}  {g['mae_b']:>8.3f}  {g['mae_avg']:>8.3f}")
        print(f"  {'Naive baseline MAE':<28} {NAIVE_XG:>8.2f}  {NAIVE_XG:>8.2f}  {NAIVE_XG:>8.2f}")
        print(f"  {'RMSE':<28} {g['rmse_a']:>8.3f}  {g['rmse_b']:>8.3f}")
        print(f"  {'Total goals MAE':<28} {g['total_goals_mae']:>8.3f}")
        print(f"  {'Goal diff MAE':<28} {g['goal_diff_mae']:>8.3f}")
        print(f"  {'Directional acc (GD)':<28} {g['directional_accuracy']:>7.1%}")
        print(f"  {'Exact scoreline acc':<28} {g['exact_score_acc']:>7.1%}")
        print(f"  {'±1 goal both sides':<28} {g['within1_acc']:>7.1%}")
        print(f"  {'Mean scoreline log-P':<28} {g['scoreline_log_p']:>8.4f}")
        print(f"  {'Naive scoreline log-P':<28} {g['naive_log_p']:>8.4f}")
        print()
        print(f"  Bias: team_a {g['bias_a']:+.3f} goals avg  |  team_b {g['bias_b']:+.3f} goals avg")
        print(f"        Strong teams: {g['bias_strong']:+.3f}  |  Weak teams: {g['bias_weak']:+.3f}")

    # Per-match goals table
    print("\n" + "=" * 70)
    print("PER-MATCH GOALS BREAKDOWN:")
    print(print_goals_table(merged))

    # ── Save ──────────────────────────────────────────────────────────────────
    BACKTEST_DIR.mkdir(exist_ok=True)
    combined = {"outcome": outcome_metrics, "goals": goals_metrics}

    def _clean(d):
        if isinstance(d, dict):
            return {k: _clean(v) for k, v in d.items()}
        if isinstance(d, (np.float32, np.float64, np.float16)):
            return float(d)
        if isinstance(d, (np.int32, np.int64)):
            return int(d)
        return d

    with open(METRICS_FILE, "w") as f:
        json.dump(_clean(combined), f, indent=2)
    print(f"\nMetrics saved -> {METRICS_FILE}")

    # Per-match RPS CSV for Streamlit
    upsets_df = detect_upsets(merged)
    rps_path  = BACKTEST_DIR / "match_rps_2026.csv"
    upsets_df.to_csv(rps_path, index=False)
    print(f"Per-match RPS -> {rps_path}")


if __name__ == "__main__":
    main()
