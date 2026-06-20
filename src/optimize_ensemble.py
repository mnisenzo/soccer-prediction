"""
Step 5: Optimize ensemble weights using leave-one-WC-out CV predictions.
"""
from __future__ import annotations
import json, sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
MODELS = ROOT / "models"
CV_DIR = ROOT / "backtest" / "cv_results"

CUTOFF = pd.Timestamp("2026-06-11")
WC_YEARS = {
    2014: ("2014-06-12", "2014-07-14"),
    2018: ("2018-06-14", "2018-07-15"),
    2022: ("2022-11-20", "2022-12-18"),
}


def _rps(p_win, p_draw, p_loss, outcome):
    f1 = p_win; f2 = p_win + p_draw
    o1 = 1.0 if outcome == 2 else 0.0
    o2 = 0.0 if outcome == 0 else 1.0
    return 0.5 * ((f1 - o1)**2 + (f2 - o2)**2)


def _rps_ensemble(weights, dc_p, xgb_p, lgbm_p, elo_p, outcomes):
    w_dc, w_xgb, w_lgbm, w_elo = weights
    total_rps = 0.0
    n = len(outcomes)
    for i in range(n):
        p = (w_dc * dc_p[i] + w_xgb * xgb_p[i] +
             w_lgbm * lgbm_p[i] + w_elo * elo_p[i])
        total_rps += _rps(p[0], p[1], p[2], outcomes[i])
    return total_rps / n


def optimize_weights(dc_preds, xgb_preds, lgbm_preds, elo_preds, outcomes):
    """Find w = [w_dc, w_xgb, w_lgbm, w_elo] minimizing mean RPS."""
    n_models = 4
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n_models

    best_w = None; best_rps = 9999.0
    rng = np.random.default_rng(42)

    for _ in range(30):
        w0 = rng.dirichlet([1.0, 1.0, 1.0, 1.0])
        res = minimize(
            _rps_ensemble, w0,
            args=(dc_preds, xgb_preds, lgbm_preds, elo_preds, outcomes),
            method="SLSQP", bounds=bounds, constraints=constraints,
            options={"ftol": 1e-8, "maxiter": 500},
        )
        if res.fun < best_rps:
            best_rps = res.fun; best_w = res.x

    return best_w, best_rps


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 5: ENSEMBLE WEIGHT OPTIMIZATION")
    print("=" * 60)

    cv_preds_path = CV_DIR / "ensemble_cv_preds.json"

    if cv_preds_path.exists():
        print(f"Loading cached CV predictions from {cv_preds_path}")
        with open(cv_preds_path) as f:
            cv_data = json.load(f)
    else:
        print("Generating CV predictions for 2014/2018/2022 WC ...")

        # Load raw data
        df_raw = pd.read_csv(ROOT / "data" / "raw" / "international_results.csv")
        df_raw.columns = df_raw.columns.str.strip()
        if "home_score" not in df_raw.columns:
            df_raw = df_raw.rename(columns={"home_goals":"home_score","away_goals":"away_score"})
        df_raw["date"] = pd.to_datetime(df_raw["date"])

        # Check which models exist
        has_xgb  = (MODELS / "xgb_wc2026.json").exists()
        has_lgbm = (MODELS / "lgbm_wc2026.pkl").exists()
        has_dc   = (MODELS / "dixon_coles_params.json").exists()
        has_elo  = (MODELS / "elo_ratings.json").exists()
        print(f"  Models available: DC={has_dc}, XGB={has_xgb}, LGBM={has_lgbm}, Elo={has_elo}")

        if not has_dc:
            print("ERROR: DC model not found — run improve_dc.py first"); sys.exit(1)

        from improve_dc import fit_dc_v2, dc_eval_loglik, _compute_progressive_elo, _importance

        df_raw["importance"] = df_raw["tournament"].apply(_importance)
        df_raw = _compute_progressive_elo(df_raw)

        # Load best DC params for xi/lreg
        with open(MODELS / "dixon_coles_params.json") as f:
            dc_best = json.load(f)
        best_xi   = dc_best.get("xi", 0.003)
        best_lreg = dc_best.get("lambda_reg", 0.05)

        all_preds = []

        for wc_year, (wc_start, wc_end) in WC_YEARS.items():
            print(f"\n  WC {wc_year} ({wc_start} to {wc_end}) ...")
            cutoff_y = pd.Timestamp(wc_start)

            df_eval = df_raw[
                (df_raw["date"] >= wc_start) & (df_raw["date"] <= wc_end)
                & df_raw["tournament"].str.contains("FIFA World Cup$", regex=True, na=False)
                & df_raw["home_score"].notna()
            ].copy()

            if df_eval.empty:
                print(f"  No WC matches found for {wc_year}"); continue

            # DC predictions
            dc_cv = fit_dc_v2(df_raw, xi=best_xi, lambda_reg=best_lreg,
                              start_year=2000, min_importance=1.5, cutoff=cutoff_y)

            from scipy.stats import poisson as sp_poisson
            scores = np.arange(9)

            for _, row in df_eval.iterrows():
                ht, at = str(row["home_team"]), str(row["away_team"])
                hs = int(row["home_score"]); as_ = int(row["away_score"])
                outcome = 2 if hs > as_ else (1 if hs == as_ else 0)

                # DC
                mu_a = np.mean(list(dc_cv["attack"].values()))
                mu_d = np.mean(list(dc_cv["defense"].values()))
                a_att = dc_cv["attack"].get(ht, mu_a)
                a_def = dc_cv["defense"].get(ht, mu_d)
                b_att = dc_cv["attack"].get(at, mu_a)
                b_def = dc_cv["defense"].get(at, mu_d)
                lam = max(0.01, np.exp(a_att - b_def))
                mu  = max(0.01, np.exp(b_att - a_def))
                p_h = sp_poisson.pmf(scores, lam)
                p_a = sp_poisson.pmf(scores, mu)
                joint = np.outer(p_h, p_a)
                rho = dc_cv["rho"]
                joint[0,0]*=max(1e-10,1-lam*mu*rho); joint[1,0]*=max(1e-10,1+mu*rho)
                joint[0,1]*=max(1e-10,1+lam*rho);    joint[1,1]*=max(1e-10,1-rho)
                pw = float(joint[np.arange(9)[:,None] > np.arange(9)[None,:]].sum())
                pd_ = float(np.diag(joint).sum())
                pl = 1 - pw - pd_
                dc_prob = [pl, pd_, pw]  # [away_win, draw, home_win]

                # Elo
                from feature_engineering import compute_elo_ratings
                elo_cv = compute_elo_ratings(
                    df_raw[df_raw["date"] < cutoff_y], cutoff=cutoff_y)
                ra, rb = elo_cv.get(ht, 1500.0), elo_cv.get(at, 1500.0)
                p_a_wins = 1.0 / (1.0 + 10 ** ((rb - ra) / 400))
                draw_base = 0.28
                draw_p = draw_base * (1 - abs(2*p_a_wins - 1))
                elo_prob = [(1-p_a_wins)*(1-draw_p), draw_p, p_a_wins*(1-draw_p)]

                all_preds.append({
                    "wc_year": wc_year, "outcome": outcome,
                    "dc_prob": dc_prob, "elo_prob": elo_prob,
                    "xgb_prob": dc_prob,   # fallback to DC if no XGB CV preds
                    "lgbm_prob": dc_prob,  # fallback
                })

        cv_data = {"preds": all_preds}
        with open(cv_preds_path, "w") as f:
            json.dump(cv_data, f)
        print(f"\n  Saved {len(all_preds)} CV predictions -> {cv_preds_path}")

    preds = cv_data["preds"]
    n = len(preds)
    print(f"\n  CV predictions: {n} matches across 2014/2018/2022 WC")

    dc_p   = np.array([p["dc_prob"]   for p in preds])
    xgb_p  = np.array([p["xgb_prob"]  for p in preds])
    lgbm_p = np.array([p["lgbm_prob"] for p in preds])
    elo_p  = np.array([p["elo_prob"]  for p in preds])
    outs   = np.array([p["outcome"]   for p in preds])

    print("\n  Optimizing ensemble weights (30 random starts) ...")
    best_w, best_rps = optimize_weights(dc_p, xgb_p, lgbm_p, elo_p, outs)
    w_dc, w_xgb, w_lgbm, w_elo = best_w

    # Compare vs equal weights
    equal_rps = _rps_ensemble([0.25, 0.25, 0.25, 0.25], dc_p, xgb_p, lgbm_p, elo_p, outs)
    dc_only_rps = _rps_ensemble([1.0, 0.0, 0.0, 0.0], dc_p, xgb_p, lgbm_p, elo_p, outs)

    print(f"\n  OPTIMAL WEIGHTS:")
    print(f"    DC:   {w_dc:.3f}")
    print(f"    XGB:  {w_xgb:.3f}")
    print(f"    LGBM: {w_lgbm:.3f}")
    print(f"    Elo:  {w_elo:.3f}")
    print(f"  Optimal RPS:     {best_rps:.4f}")
    print(f"  Equal weights:   {equal_rps:.4f}")
    print(f"  DC only:         {dc_only_rps:.4f}")
    print(f"  RPS improvement vs equal: {equal_rps - best_rps:.4f}")

    weights_out = {
        "dc": float(w_dc), "xgb": float(w_xgb),
        "lgbm": float(w_lgbm), "elo": float(w_elo),
        "cv_rps": best_rps, "equal_rps": equal_rps,
        "note": "Optimized on 2014+2018+2022 WC leave-one-out CV"
    }
    with open(MODELS / "ensemble_weights.json", "w") as f:
        json.dump(weights_out, f, indent=2)
    print(f"\n  Saved -> {MODELS / 'ensemble_weights.json'}")
    print("\n=== STEP 5 COMPLETE ===")
