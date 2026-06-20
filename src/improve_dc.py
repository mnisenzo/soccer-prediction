"""
Step 2: Improved Dixon-Coles model.
2a. xi grid search with leave-one-WC-out CV
2b. rho already free — verify
2c. Opposition-strength weighting via progressive Elo
2d. L2 regularization tuning
2e. Refit and save
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson as sp_poisson

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

CUTOFF = pd.Timestamp("2026-06-11")
TODAY  = pd.Timestamp.today()
CV_DIR = ROOT / "backtest" / "cv_results"
CV_DIR.mkdir(exist_ok=True)

WC_YEARS = {
    2014: ("2014-06-12", "2014-07-14"),
    2018: ("2018-06-14", "2018-07-15"),
    2022: ("2022-11-20", "2022-12-18"),
}

TOURNAMENT_K = {
    "FIFA World Cup": 4.0,
    "UEFA Euro": 2.0, "Copa America": 2.0,
    "Africa Cup of Nations": 2.0, "Asian Cup": 2.0,
    "CONCACAF Gold Cup": 2.0,
    "FIFA World Cup qualification": 1.5,
    "Friendly": 0.5,
}

def _importance(t: str) -> float:
    for k, v in TOURNAMENT_K.items():
        if k.lower() in str(t).lower():
            return v
    return 1.0


def _compute_progressive_elo(df: pd.DataFrame) -> pd.DataFrame:
    """Add elo_home_pre / elo_away_pre columns (Elo before each match)."""
    df = df.sort_values("date").reset_index(drop=True)
    ratings: dict[str, float] = {}
    elo_h_pre, elo_a_pre = [], []
    K_MAP = {
        "FIFA World Cup": 60, "UEFA Euro": 50,
        "Copa America": 50, "FIFA World Cup qualification": 40,
    }
    for _, row in df.iterrows():
        home, away = str(row["home_team"]), str(row["away_team"])
        rh, ra = ratings.get(home, 1500.0), ratings.get(away, 1500.0)
        elo_h_pre.append(rh)
        elo_a_pre.append(ra)
        hs = row.get("home_score"); as_ = row.get("away_score")
        if pd.isna(hs) or pd.isna(as_):
            continue
        hs, as_ = int(hs), int(as_)
        neutral = bool(row.get("neutral", False))
        hadv = 0.0 if neutral else 75.0
        tourn = str(row.get("tournament", ""))
        k = next((v for k, v in K_MAP.items() if k.lower() in tourn.lower()), 30)
        e_h = 1.0 / (1.0 + 10 ** (-((rh + hadv) - ra) / 400))
        s_h = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        ratings[home] = rh + k * (s_h - e_h)
        ratings[away] = ra + k * ((1 - s_h) - (1 - e_h))
    df["elo_home_pre"] = elo_h_pre
    df["elo_away_pre"] = elo_a_pre
    return df


def _dc_loglik_v2(
    params: np.ndarray,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
    lambda_reg: float = 0.05,
) -> float:
    attack  = np.concatenate([[0.0], params[:n_teams - 1]])
    defense = params[n_teams - 1: 2 * n_teams - 1]
    home_adv = params[2 * n_teams - 1]
    rho      = params[2 * n_teams]

    lam = np.exp(attack[home_idx] - defense[away_idx] + home_adv).clip(0.01)
    mu  = np.exp(attack[away_idx] - defense[home_idx]).clip(0.01)

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

    ll_home = x * np.log(lam) - lam - gammaln(x + 1)
    ll_away = y * np.log(mu)  - mu  - gammaln(y + 1)
    ll = weights @ (np.log(tau) + ll_home + ll_away)

    atk_free = params[:n_teams - 1]
    l2 = lambda_reg * (np.sum(atk_free ** 2) + np.sum(defense ** 2))
    return -float(ll) + l2


def fit_dc_v2(df_all: pd.DataFrame, xi: float, lambda_reg: float,
              start_year: int = 2000, min_importance: float = 1.5,
              cutoff: pd.Timestamp = CUTOFF) -> dict:
    d = df_all[
        (df_all["date"] >= pd.Timestamp(f"{start_year}-01-01"))
        & (df_all["date"] < cutoff)
        & (df_all["importance"] >= min_importance)
        & df_all["home_score"].notna()
        & df_all["away_score"].notna()
    ].copy()

    ref_date = cutoff
    d["days_ago"] = (ref_date - d["date"]).dt.days
    d["time_w"] = np.exp(-xi * d["days_ago"]) * d["importance"]

    # Opposition-strength weighting
    avg_elo = (d["elo_home_pre"] + d["elo_away_pre"]) / 2.0
    d["strength_w"] = 1.0 / (1.0 + np.exp(-(avg_elo - 1500.0) / 200.0))
    d["weight"] = d["time_w"] * d["strength_w"]

    teams = sorted(set(d["home_team"]) | set(d["away_team"]))
    tidx  = {t: i for i, t in enumerate(teams)}
    n_t   = len(teams)

    hi = d["home_team"].map(tidx).values.astype(int)
    ai = d["away_team"].map(tidx).values.astype(int)
    hg = d["home_score"].values.astype(int)
    ag = d["away_score"].values.astype(int)
    w  = d["weight"].values

    n_params = (n_t - 1) + n_t + 2
    x0 = np.zeros(n_params)
    x0[2 * n_t - 1] = 0.1
    x0[2 * n_t]     = -0.05

    bounds = (
        [(-3.0, 3.0)] * (n_t - 1)
        + [(-3.0, 3.0)] * n_t
        + [(0.0, 1.0)]
        + [(-0.5, 0.3)]
    )

    result = minimize(
        _dc_loglik_v2, x0,
        args=(hi, ai, hg, ag, w, n_t, lambda_reg),
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 1500, "maxfun": n_params * 1000,
                 "ftol": 1e-8, "gtol": 1e-5},
    )

    p = result.x
    atk = np.concatenate([[0.0], p[:n_t - 1]])
    dfn = p[n_t - 1: 2 * n_t - 1]
    return {
        "teams":  teams,
        "attack":  {t: float(atk[i]) for i, t in enumerate(teams)},
        "defense": {t: float(dfn[i]) for i, t in enumerate(teams)},
        "home_advantage": float(p[2 * n_t - 1]),
        "rho":            float(p[2 * n_t]),
        "xi":             xi,
        "lambda_reg":     lambda_reg,
    }


def dc_eval_loglik(dc: dict, df_wc: pd.DataFrame) -> float:
    atk = dc["attack"]; dfn = dc["defense"]
    rho = dc["rho"]
    mu_a = float(np.mean(list(atk.values())))
    mu_d = float(np.mean(list(dfn.values())))
    scores = np.arange(9)
    log_p_vals = []
    for _, row in df_wc.iterrows():
        ht = str(row["home_team"]); at = str(row["away_team"])
        hs = int(row["home_score"]); as_ = int(row["away_score"])
        lam = max(0.01, float(np.exp(atk.get(ht, mu_a) - dfn.get(at, mu_d))))
        mu  = max(0.01, float(np.exp(atk.get(at, mu_a) - dfn.get(ht, mu_d))))
        p_h = sp_poisson.pmf(scores, lam)
        p_a = sp_poisson.pmf(scores, mu)
        joint = np.outer(p_h, p_a)
        joint[0,0] *= max(1e-10, 1 - lam*mu*rho)
        joint[1,0] *= max(1e-10, 1 + mu*rho)
        joint[0,1] *= max(1e-10, 1 + lam*rho)
        joint[1,1] *= max(1e-10, 1 - rho)
        joint = joint.clip(1e-12)
        si = min(hs, 8); sj = min(as_, 8)
        log_p_vals.append(np.log(float(joint[si, sj])))
    return float(np.mean(log_p_vals)) if log_p_vals else -999.0


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 2: IMPROVED DIXON-COLES")
    print("=" * 60)

    df_raw = pd.read_csv(ROOT / "data" / "raw" / "international_results.csv")
    df_raw.columns = df_raw.columns.str.strip()
    if "home_score" not in df_raw.columns and "home_goals" in df_raw.columns:
        df_raw = df_raw.rename(columns={"home_goals":"home_score","away_goals":"away_score"})
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    df_raw["importance"] = df_raw["tournament"].apply(_importance)

    print("Computing progressive Elo ...")
    t0 = time.time()
    df_raw = _compute_progressive_elo(df_raw)
    print(f"  Done in {time.time()-t0:.1f}s  ({len(df_raw)} rows)")

    cv_cache = CV_DIR / "dc_cv_results.json"
    if cv_cache.exists():
        print(f"\n[2a] Loading cached CV: {cv_cache}")
        with open(cv_cache) as f:
            cv_results = json.load(f)
    else:
        xi_grid  = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]
        lam_grid = [0.01, 0.05, 0.1, 0.2]
        cv_results = []
        total = len(xi_grid) * len(lam_grid) * len(WC_YEARS)
        print(f"\n[2a+2d] CV grid: {len(xi_grid)} xi x {len(lam_grid)} lreg x {len(WC_YEARS)} WC = {total} fits")

        for xi in xi_grid:
            for lreg in lam_grid:
                wc_logps = []
                for wc_year, (wc_start, wc_end) in WC_YEARS.items():
                    cutoff_y = pd.Timestamp(wc_start)
                    df_eval = df_raw[
                        (df_raw["date"] >= wc_start) & (df_raw["date"] <= wc_end)
                        & df_raw["tournament"].str.contains("FIFA World Cup$", regex=True, na=False)
                        & df_raw["home_score"].notna()
                    ].copy()
                    if df_eval.empty:
                        continue
                    dc_cv = fit_dc_v2(df_raw, xi=xi, lambda_reg=lreg,
                                      start_year=2000, min_importance=1.5,
                                      cutoff=cutoff_y)
                    logp = dc_eval_loglik(dc_cv, df_eval)
                    wc_logps.append(logp)
                    print(f"  xi={xi:.3f} lreg={lreg:.2f} WC{wc_year}: {logp:.4f}")
                mean_logp = float(np.mean(wc_logps)) if wc_logps else -999.0
                cv_results.append({"xi": xi, "lambda_reg": lreg, "mean_log_p": mean_logp})
                print(f"  -> MEAN={mean_logp:.4f}\n")

        with open(cv_cache, "w") as f:
            json.dump(cv_results, f, indent=2)
        print(f"  CV results saved -> {cv_cache}")

    best = max(cv_results, key=lambda r: r["mean_log_p"])
    print(f"\n[2a] BEST: xi={best['xi']:.3f}, lambda_reg={best['lambda_reg']:.3f}, mean_log_P={best['mean_log_p']:.4f}")
    print("\n  Full CV table (best per xi):")
    for xi_val in sorted(set(r["xi"] for r in cv_results)):
        bests = [r for r in cv_results if r["xi"] == xi_val]
        b = max(bests, key=lambda x: x["mean_log_p"])
        print(f"  xi={xi_val:.3f}  best_lreg={b['lambda_reg']:.2f}  log_P={b['mean_log_p']:.4f}")

    print("\n[2b] rho: free param in [-0.5, 0.3] — confirmed")
    print("[2c] Opposition-strength weighting: sigmoid((avg_elo-1500)/200) — ACTIVE")
    print(f"[2d] L2 lambda_reg={best['lambda_reg']:.3f} (CV-tuned)")

    print(f"\n[2e] Fitting final model: xi={best['xi']:.3f}, lreg={best['lambda_reg']:.3f}")
    t0 = time.time()
    dc_final = fit_dc_v2(df_raw, xi=best["xi"], lambda_reg=best["lambda_reg"],
                         start_year=2000, min_importance=1.5, cutoff=CUTOFF)
    dc_final["trained_at"] = pd.Timestamp.utcnow().isoformat()
    print(f"  Fit in {time.time()-t0:.1f}s | rho={dc_final['rho']:.4f} | home_adv={dc_final['home_advantage']:.4f}")
    print(f"  Teams: {len(dc_final['attack'])}")

    atk = dc_final["attack"]
    sorted_atk = sorted(atk.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 15 attack: {[(t, round(v,3)) for t,v in sorted_atk[:15]]}")
    print(f"  Bot 10 attack: {[(t, round(v,3)) for t,v in sorted_atk[-10:]]}")

    top5 = {t for t,_ in sorted_atk[:5]}
    expected_top5 = {"France","Brazil","Argentina","Spain","England"}
    print(f"\n  Top-5 overlap with {expected_top5}: {top5 & expected_top5}")

    key_teams = ["France","Brazil","Argentina","Spain","England","Germany",
                 "Norway","Jordan","Qatar","Panama","Curacao","Iraq","Czechia","USA","Cabo Verde"]
    print("\n  Key team attack params (WC 2026):")
    for t in key_teams:
        v = atk.get(t, atk.get("United States" if t=="USA" else
                               "Czech Republic" if t=="Czechia" else
                               "Cape Verde" if t=="Cabo Verde" else t, "MISSING"))
        print(f"    {t}: {round(v,3) if isinstance(v,float) else v}")

    out = ROOT / "models" / "dixon_coles_params.json"
    with open(out, "w") as f:
        json.dump(dc_final, f, indent=2)
    print(f"\n  Saved -> {out}")
    print("\n=== STEP 2 COMPLETE ===")
