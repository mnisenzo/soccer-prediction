"""
Generate a markdown backtest report for WC 2026.

Usage:
    python backtest/generate_report.py
Output:
    backtest/report_2026.md
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backtest"))

from constants import BACKTEST_DIR

PRED_FILE = BACKTEST_DIR / "pretournament_predictions.csv"
RESULTS_FILE = BACKTEST_DIR / "completed_results.csv"
METRICS_FILE = BACKTEST_DIR / "metrics_2026.json"
MATCH_RPS_FILE = BACKTEST_DIR / "match_rps_2026.csv"
REPORT_FILE = BACKTEST_DIR / "report_2026.md"


MODEL_LABELS = {
    "dc": "Dixon-Coles",
    "xgb": "XGBoost",
    "elo": "Elo",
    "ens": "Ensemble",
    "baseline_historical": "Historical WC baseline",
    "baseline_uniform": "Uniform (1/3 each)",
}


def _star(rps_mean: float, baseline: float) -> str:
    diff = rps_mean - baseline
    if diff < -0.02:
        return " **[beats baseline]**"
    if diff > 0.02:
        return " *(worse than baseline)*"
    return ""


def generate_report() -> str:
    """Build the markdown report string."""
    lines = []
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines.append(f"# WC 2026 Backtest Report")
    lines.append(f"*Generated {now}*")
    lines.append("")

    # Load data
    if not METRICS_FILE.exists():
        lines.append("**ERROR: Run `python backtest/evaluate_2026.py` first to generate metrics.**")
        return "\n".join(lines)

    with open(METRICS_FILE) as f:
        metrics = json.load(f)

    n_matches = metrics.get("ens", metrics.get("dc", {})).get("n", 0)
    lines.append(f"**Matches evaluated:** {n_matches}")
    lines.append(
        f"> **Small sample warning**: {n_matches} matches. "
        "Bootstrap CIs are wide — interpret with caution."
    )
    lines.append("")

    # ── Summary metrics table ──────────────────────────────────────────────────
    lines.append("## Summary Metrics")
    lines.append("")
    lines.append("| Model | RPS (lower=better) | 95% CI | Log-Loss | Accuracy |")
    lines.append("|-------|-------------------|--------|----------|----------|")

    baseline_rps = metrics.get("baseline_historical", {}).get("rps_mean", 0.25)
    order = ["ens", "dc", "xgb", "elo", "baseline_historical", "baseline_uniform"]
    for k in order:
        if k not in metrics:
            continue
        m = metrics[k]
        label = MODEL_LABELS.get(k, k)
        rps_m = m.get("rps_mean", 0)
        ci_lo = m.get("rps_ci_lo", rps_m)
        ci_hi = m.get("rps_ci_hi", rps_m)
        ll = m.get("log_loss", 0)
        acc = m.get("accuracy", 0)
        star = _star(rps_m, baseline_rps) if k not in ("baseline_historical", "baseline_uniform") else ""
        lines.append(
            f"| {label}{star} | {rps_m:.4f} | [{ci_lo:.3f}, {ci_hi:.3f}] | {ll:.4f} | {acc:.1%} |"
        )

    lines.append("")
    lines.append(
        "*RPS = Ranked Probability Score (lower is better). "
        "Historical baseline: 45% W / 27% D / 28% L.*"
    )
    lines.append("")

    # ── Dixon-Coles xG section ─────────────────────────────────────────────────
    if "dc" in metrics and "xg_mae_a" in metrics["dc"]:
        d = metrics["dc"]
        lines.append("## Dixon-Coles Expected Goals")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Goals-for MAE | {d['xg_mae_a']:.2f} |")
        lines.append(f"| Goals-against MAE | {d['xg_mae_b']:.2f} |")
        lines.append(f"| Correct goal-scorer direction | {d['xg_direction_correct']}/{d['xg_n']} ({d['xg_direction_correct']/d['xg_n']:.0%}) |")
        lines.append("")

    # ── Calibration note ──────────────────────────────────────────────────────
    lines.append("## Calibration")
    lines.append("")
    lines.append(
        f"Calibration analysis requires binning probabilities. "
        f"With only {n_matches} matches, each bucket has very few samples — "
        "formal calibration testing is deferred until 30+ matches are available."
    )
    lines.append("")

    # ── Upset detection ───────────────────────────────────────────────────────
    if MATCH_RPS_FILE.exists():
        match_df = pd.read_csv(MATCH_RPS_FILE)
        upsets = match_df[match_df["is_upset"] == True].sort_values("rps", ascending=False)
        correct = match_df[match_df["is_upset"] == False].sort_values("rps")

        lines.append(f"## Upset Detection")
        lines.append("")
        n_upsets = len(upsets)
        lines.append(f"**{n_upsets} upset(s)** out of {len(match_df)} matches (model favourite was wrong).")
        lines.append("")

        if not upsets.empty:
            lines.append("| Match | Score | Actual | Model Predicted | Model p(actual) | RPS |")
            lines.append("|-------|-------|--------|----------------|-----------------|-----|")
            for _, u in upsets.iterrows():
                ta = str(u["team_a"])[:12]
                tb = str(u["team_b"])[:12]
                p_act = float(u["prob_of_actual"])
                r = float(u["rps"])
                lines.append(
                    f"| {ta} vs {tb} | {u['score']} | {u['outcome_label']} "
                    f"| {u['predicted_label']} | {p_act:.0%} | {r:.3f} |"
                )
            lines.append("")

        lines.append("### Biggest Surprises (highest RPS = model was most wrong)")
        lines.append("")
        for _, u in match_df.sort_values("rps", ascending=False).head(5).iterrows():
            correct_lbl = "WRONG" if u["is_upset"] else "correct"
            p_act = float(u["prob_of_actual"])
            lines.append(
                f"- **{u['team_a']} vs {u['team_b']}** ({u['group']}): "
                f"Score {u['score']}, {u['outcome_label']}. "
                f"Model gave {p_act:.0%} — {correct_lbl}. RPS={float(u['rps']):.3f}"
            )
        lines.append("")

        lines.append("### Best Predictions (lowest RPS = model was most confident and correct)")
        lines.append("")
        for _, u in match_df.sort_values("rps").head(5).iterrows():
            p_act = float(u["prob_of_actual"])
            lines.append(
                f"- **{u['team_a']} vs {u['team_b']}** ({u['group']}): "
                f"Score {u['score']}, {u['outcome_label']}. "
                f"p={p_act:.0%}. RPS={float(u['rps']):.3f}"
            )
        lines.append("")

    # ── Match-by-match ─────────────────────────────────────────────────────────
    if MATCH_RPS_FILE.exists():
        match_df = pd.read_csv(MATCH_RPS_FILE)
        lines.append("## Match-by-Match Results")
        lines.append("")
        lines.append(
            "| Date | Group | Match | Score | Actual | Model | Ens p(A) | p(D) | p(B) | p(actual) | RPS |"
        )
        lines.append(
            "|------|-------|-------|-------|--------|-------|----------|------|------|-----------|-----|"
        )
        for _, u in match_df.sort_values("date").iterrows():
            correct = "✓" if not u["is_upset"] else "✗"
            pa = float(u.get("ens_prob_a", 0))
            pd_ = float(u.get("ens_draw", 0))
            pb = float(u.get("ens_prob_b", 0))
            p_act = float(u["prob_of_actual"])
            r = float(u["rps"])
            date_s = str(u["date"])[:10]
            lines.append(
                f"| {date_s} | {u['group']} | {u['team_a']} v {u['team_b']} "
                f"| {u['score']} | {u['outcome_label']} | {correct} {u['predicted_label']} "
                f"| {pa:.0%} | {pd_:.0%} | {pb:.0%} | {p_act:.0%} | {r:.3f} |"
            )
        lines.append("")

    # ── Kalshi comparison ──────────────────────────────────────────────────────
    kalshi_file = BACKTEST_DIR / "kalshi_comparison_2026.csv"
    lines.append("## Kalshi Market Comparison")
    lines.append("")
    if kalshi_file.exists():
        kdf = pd.read_csv(kalshi_file)
        available = kdf[kdf["kalshi_available"] == True]
        if not available.empty:
            lines.append(f"Pre-match Kalshi odds available for {len(available)}/{len(kdf)} matches.")
            lines.append("")
            lines.append(
                "| Match | Outcome | Model RPS | Kalshi RPS | Diff (model - Kalshi) |"
            )
            lines.append("|-------|---------|-----------|------------|----------------------|")
            for _, r in available.iterrows():
                out = {2: "A wins", 1: "Draw", 0: "B wins"}.get(r["outcome"], "?")
                rk = r.get("rps_kalshi")
                rd = r.get("rps_diff")
                rk_s = f"{rk:.3f}" if pd.notna(rk) else "N/A"
                rd_s = f"{rd:+.3f}" if pd.notna(rd) else "N/A"
                lines.append(
                    f"| {r['team_a']} v {r['team_b']} ({r['group']}) "
                    f"| {out} | {r['rps_model']:.3f} | {rk_s} | {rd_s} |"
                )
            lines.append("")
        else:
            lines.append(
                "Pre-match Kalshi odds not available for completed matches. "
                "Settled markets reset to 0.99/0.01 post-match; "
                "Wayback Machine snapshots did not yield price data."
            )
            lines.append("")
            lines.append(
                "**Note**: Kalshi winner market live odds (retrieved June 19, 2026): "
                "France 17.5%, Spain 12.4%, England 12.1%, Argentina 10.1%. "
                "Model disagrees most on Mexico (+14.9pp) and France (-14.1pp)."
            )
    else:
        lines.append(
            "Run `python backtest/kalshi_2026_backtest.py` to generate Kalshi comparison."
        )
    lines.append("")

    # ── Methodology notes ─────────────────────────────────────────────────────
    lines.append("## Methodology")
    lines.append("")
    lines.append("- **RPS** (Ranked Probability Score): primary metric. Formula: `0.5 * [(F1-O1)² + (F1+F2 - O1-O2)²]`")
    lines.append("- **Ensemble**: Dixon-Coles 66.7% + Elo 33.3% (XGBoost not trained in this run)")
    lines.append("- **Predictions frozen pre-tournament** — immutable, not updated as matches complete")
    lines.append("- **Historical baseline**: 45%/27%/28% W/D/L from 2010-2022 WC group stages")
    lines.append("- **Bootstrap CI**: 10,000 resamples with seed=42")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    BACKTEST_DIR.mkdir(exist_ok=True)

    if not METRICS_FILE.exists():
        print("Metrics not found. Running evaluate_2026.py ...")
        import subprocess, os
        anaconda_py = r"C:\Users\nisen\anaconda3\python.exe"
        python_exe = anaconda_py if os.path.exists(anaconda_py) else sys.executable
        result = subprocess.run(
            [python_exe, str(PROJECT_ROOT / "backtest" / "evaluate_2026.py")],
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            print("evaluate_2026.py failed. Run it manually first.")
            sys.exit(1)

    report = generate_report()
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(f"Report written to {REPORT_FILE}")
    print(f"  Lines: {report.count(chr(10))}")


if __name__ == "__main__":
    main()
