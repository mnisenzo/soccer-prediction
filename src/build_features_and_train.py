"""
Steps 3+4: Feature engineering + XGBoost + LightGBM + Calibration.
"""
from __future__ import annotations
import json, sys, time, warnings, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import poisson as sp_poisson

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
PROCESSED = ROOT / "data" / "processed"
MODELS    = ROOT / "models"
RAW       = ROOT / "data" / "raw"
PROCESSED.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

CUTOFF = pd.Timestamp("2026-06-11")

CONFEDERATION_MAP = {
    "Germany":"UEFA","France":"UEFA","Spain":"UEFA","Portugal":"UEFA",
    "Netherlands":"UEFA","Belgium":"UEFA","England":"UEFA","Italy":"UEFA",
    "Croatia":"UEFA","Austria":"UEFA","Denmark":"UEFA","Switzerland":"UEFA",
    "Sweden":"UEFA","Norway":"UEFA","Poland":"UEFA","Czech Republic":"UEFA",
    "Czechia":"UEFA","Serbia":"UEFA","Scotland":"UEFA","Turkey":"UEFA",
    "Bosnia and Herzegovina":"UEFA","Slovakia":"UEFA","Hungary":"UEFA",
    "Ukraine":"UEFA","Finland":"UEFA","Albania":"UEFA","Romania":"UEFA",
    "Brazil":"CONMEBOL","Argentina":"CONMEBOL","Colombia":"CONMEBOL",
    "Uruguay":"CONMEBOL","Ecuador":"CONMEBOL","Paraguay":"CONMEBOL",
    "Bolivia":"CONMEBOL","Venezuela":"CONMEBOL","Peru":"CONMEBOL","Chile":"CONMEBOL",
    "United States":"CONCACAF","USA":"CONCACAF","Canada":"CONCACAF","Mexico":"CONCACAF",
    "Panama":"CONCACAF","Jamaica":"CONCACAF","Costa Rica":"CONCACAF",
    "Curacao":"CONCACAF","Curaçao":"CONCACAF","Haiti":"CONCACAF",
    "Japan":"AFC","South Korea":"AFC","Korea Republic":"AFC","Iran":"AFC",
    "IR Iran":"AFC","Saudi Arabia":"AFC","Australia":"AFC","Qatar":"AFC",
    "Iraq":"AFC","Uzbekistan":"AFC","Jordan":"AFC",
    "Senegal":"CAF","Morocco":"CAF","Egypt":"CAF","Nigeria":"CAF",
    "Ivory Coast":"CAF","Cote d'Ivoire":"CAF","South Africa":"CAF","Ghana":"CAF",
    "Tunisia":"CAF","DR Congo":"CAF","Congo DR":"CAF","Mali":"CAF",
    "Cameroon":"CAF","Algeria":"CAF","Cape Verde":"CAF","Cabo Verde":"CAF",
    "New Zealand":"OFC",
}
CONF_ENC = {"UEFA":0,"CONMEBOL":1,"CONCACAF":2,"AFC":3,"CAF":4,"OFC":5,"OTHER":6}

TOURNAMENT_IMPORTANCE = {
    "FIFA World Cup": 4.0, "FIFA World Cup qualification": 1.5,
    "UEFA Euro": 2.0, "Copa America": 2.0, "Copa America": 2.0,
    "Africa Cup of Nations": 2.0, "Asian Cup": 2.0, "CONCACAF Gold Cup": 2.0,
    "Friendly": 0.5,
}
def _imp(t: str) -> float:
    for k, v in TOURNAMENT_IMPORTANCE.items():
        if k.lower() in str(t).lower(): return v
    return 1.0

def _kfac(t: str) -> float:
    kmap = {"FIFA World Cup": 60, "UEFA Euro": 50, "Copa America": 50,
            "Africa Cup": 40, "Asian Cup": 40,
            "FIFA World Cup qualification": 40, "CONCACAF Gold Cup": 35}
    for k, v in kmap.items():
        if k.lower() in str(t).lower(): return float(v)
    return 30.0

def _conf(t: str) -> str:
    return CONFEDERATION_MAP.get(t, "OTHER")


def build_feature_matrix(df_all: pd.DataFrame, start_year: int = 2000,
                          min_importance: float = 1.0) -> pd.DataFrame:
    df = df_all[df_all["home_score"].notna() & df_all["away_score"].notna()].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] < CUTOFF].sort_values("date").reset_index(drop=True)
    df["importance"] = df["tournament"].apply(_imp)
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    print(f"  Total pre-cutoff matches with scores: {len(df)}")

    # 1. Progressive Elo
    print("  Computing progressive Elo ...")
    ratings: dict[str, float] = {}
    elo_h_pre, elo_a_pre = [], []
    for _, row in df.iterrows():
        home, away = str(row["home_team"]), str(row["away_team"])
        rh, ra = ratings.get(home, 1500.0), ratings.get(away, 1500.0)
        elo_h_pre.append(rh); elo_a_pre.append(ra)
        neutral = bool(row.get("neutral", False))
        hadv = 0.0 if neutral else 75.0
        k = _kfac(str(row.get("tournament", "")))
        e_h = 1.0 / (1.0 + 10 ** (-((rh + hadv) - ra) / 400))
        hs, as_ = row["home_score"], row["away_score"]
        s_h = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        ratings[home] = rh + k * (s_h - e_h)
        ratings[away] = ra + k * ((1 - s_h) - (1 - e_h))
    df["elo_h"] = elo_h_pre
    df["elo_a"] = elo_a_pre
    df["elo_diff"] = df["elo_h"] - df["elo_a"]

    # 2. Rolling form via groupby + merge_asof (vectorized)
    print("  Computing rolling form (vectorized) ...")
    home_df = df[["date","home_team","home_score","away_score"]].copy()
    home_df.columns = ["date","team","gf","ga"]
    away_df = df[["date","away_team","away_score","home_score"]].copy()
    away_df.columns = ["date","team","gf","ga"]
    long = pd.concat([home_df, away_df], ignore_index=True)
    long["win"]  = (long["gf"] > long["ga"]).astype(float)
    long["draw"] = (long["gf"] == long["ga"]).astype(float)
    long["loss"] = (long["gf"] < long["ga"]).astype(float)
    long["pts"]  = long["win"] * 3 + long["draw"]
    long = long.sort_values(["team","date"]).reset_index(drop=True)

    def _roll(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").copy()
        for c in ["win","draw","loss","gf","ga","pts"]:
            g[f"r10_{c}"] = g[c].shift(1).rolling(10, min_periods=1).mean()
        g["r10_gd"] = g["r10_gf"].fillna(0) - g["r10_ga"].fillna(0)
        g["last_date"] = g["date"].shift(1)
        return g

    long_form = long.groupby("team", group_keys=False).apply(_roll)
    FORM_COLS = ["r10_win","r10_draw","r10_loss","r10_gf","r10_ga","r10_gd","r10_pts"]

    hf = (long_form[["date","team"] + FORM_COLS + ["last_date"]]
          .rename(columns={"team":"home_team", "last_date":"h_last_date",
                           **{c: f"h_{c}" for c in FORM_COLS}})
          .sort_values("date"))
    af = (long_form[["date","team"] + FORM_COLS + ["last_date"]]
          .rename(columns={"team":"away_team", "last_date":"a_last_date",
                           **{c: f"a_{c}" for c in FORM_COLS}})
          .sort_values("date"))

    cutoff_start = pd.Timestamp(f"{start_year}-01-01")
    train_df = (df[(df["date"] >= cutoff_start) & (df["importance"] >= min_importance)]
                .copy().sort_values("date").reset_index(drop=True))
    print(f"  Training window: {len(train_df)} matches")

    train_df = pd.merge_asof(train_df, hf, on="date", by="home_team", direction="backward")
    train_df = pd.merge_asof(train_df, af, on="date", by="away_team", direction="backward")

    # Vectorized days rest
    train_df["h_days_rest"] = (
        (train_df["date"] - pd.to_datetime(train_df["h_last_date"]))
        .dt.days.fillna(90).clip(0, 365).astype(int)
    )
    train_df["a_days_rest"] = (
        (train_df["date"] - pd.to_datetime(train_df["a_last_date"]))
        .dt.days.fillna(90).clip(0, 365).astype(int)
    )
    defaults_h = {"h_r10_win":0.4,"h_r10_draw":0.25,"h_r10_loss":0.35,
                  "h_r10_gf":1.3,"h_r10_ga":1.3,"h_r10_gd":0.0,"h_r10_pts":1.45}
    defaults_a = {"a_r10_win":0.4,"a_r10_draw":0.25,"a_r10_loss":0.35,
                  "a_r10_gf":1.3,"a_r10_ga":1.3,"a_r10_gd":0.0,"a_r10_pts":1.45}
    for col, val in {**defaults_h, **defaults_a}.items():
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna(val)

    # 3. Pre-build H2H dict: O(n) build, O(1) lookup
    print("  Building H2H dict ...")
    h2h_dict: dict[tuple, list] = {}
    for _, row in df.iterrows():
        home, away = str(row["home_team"]), str(row["away_team"])
        pair = (min(home, away), max(home, away))
        h2h_dict.setdefault(pair, []).append(
            (row["date"], home, row["home_score"], row["away_score"]))

    # 4. Pre-build WC history via incremental scan: O(WC_matches)
    print("  Building WC history ...")
    wc_matches = (df[df["tournament"].str.contains("FIFA World Cup$", regex=True, na=False)]
                  .sort_values("date"))
    wc_matches = wc_matches.assign(wc_year=wc_matches["date"].dt.year)
    wc_years_sorted = sorted(wc_matches["wc_year"].unique())

    wc_apps_accum: dict[str, set] = {}
    wc_wins_accum: dict[str, int] = {}
    wc_lookup: dict[tuple, dict] = {}

    for yr in wc_years_sorted:
        yr_m = wc_matches[wc_matches["wc_year"] == yr]
        teams_yr = set(yr_m["home_team"]) | set(yr_m["away_team"])
        for team in teams_yr:
            wc_lookup[(team, yr)] = {
                "wc_apps": len(wc_apps_accum.get(team, set())),
                "wc_wins": wc_wins_accum.get(team, 0),
            }
        for _, r in yr_m.iterrows():
            ht, at = str(r["home_team"]), str(r["away_team"])
            hs, as_ = r["home_score"], r["away_score"]
            wc_apps_accum.setdefault(ht, set()).add(yr)
            wc_apps_accum.setdefault(at, set()).add(yr)
            if hs > as_: wc_wins_accum[ht] = wc_wins_accum.get(ht, 0) + 1
            elif as_ > hs: wc_wins_accum[at] = wc_wins_accum.get(at, 0) + 1

    def get_wc_stats(team: str, match_year: int) -> dict:
        for yr in reversed(wc_years_sorted):
            if yr <= match_year and (team, yr) in wc_lookup:
                return wc_lookup[(team, yr)]
        return {"wc_apps": 0, "wc_wins": 0}

    # 5. Final feature assembly: loop only for H2H + WC history (O(n * k), k<=5)
    print(f"  Assembling features (H2H + WC history) for {len(train_df)} rows ...")
    h2h_hw = []; h2h_dw = []; h2h_aw = []
    h_wca = []; h_wcw = []; a_wca = []; a_wcw = []
    hce = []; ace = []; sc = []

    for _, row in train_df.iterrows():
        home, away = str(row["home_team"]), str(row["away_team"])
        dt, yr = row["date"], row["date"].year

        pair = (min(home, away), max(home, away))
        past_h2h = [(d, ht, hs, as_) for d, ht, hs, as_
                    in h2h_dict.get(pair, []) if d < dt][-5:]
        hw = dw = aw = 0
        for _, ht2, hs2, as_2 in past_h2h:
            gf, ga = (hs2, as_2) if ht2 == home else (as_2, hs2)
            if gf > ga: hw += 1
            elif gf == ga: dw += 1
            else: aw += 1
        h2h_hw.append(hw); h2h_dw.append(dw); h2h_aw.append(aw)

        hs_wc = get_wc_stats(home, yr); as_wc = get_wc_stats(away, yr)
        h_wca.append(hs_wc["wc_apps"]); h_wcw.append(hs_wc["wc_wins"])
        a_wca.append(as_wc["wc_apps"]); a_wcw.append(as_wc["wc_wins"])

        hc, ac = _conf(home), _conf(away)
        hce.append(CONF_ENC.get(hc, 6)); ace.append(CONF_ENC.get(ac, 6))
        sc.append(int(hc == ac))

    train_df["h2h_home_wins"]  = h2h_hw
    train_df["h2h_draws"]      = h2h_dw
    train_df["h2h_away_wins"]  = h2h_aw
    train_df["wc_apps_home"]   = h_wca
    train_df["wc_wins_home"]   = h_wcw
    train_df["wc_apps_away"]   = a_wca
    train_df["wc_wins_away"]   = a_wcw
    train_df["conf_home_enc"]  = hce
    train_df["conf_away_enc"]  = ace
    train_df["same_confederation"] = sc
    if "neutral" in train_df.columns:
        train_df["is_neutral"] = train_df["neutral"].fillna(False).astype(int)
    else:
        train_df["is_neutral"] = 0
    train_df["target"] = np.where(train_df["home_score"] > train_df["away_score"], 2,
                         np.where(train_df["home_score"] == train_df["away_score"], 1, 0))
    return train_df


FEATURE_COLS = [
    "elo_diff", "elo_h", "elo_a",
    "h_r10_win", "h_r10_draw", "h_r10_loss", "h_r10_gf", "h_r10_ga",
    "h_r10_gd", "h_r10_pts", "h_days_rest",
    "a_r10_win", "a_r10_draw", "a_r10_loss", "a_r10_gf", "a_r10_ga",
    "a_r10_gd", "a_r10_pts", "a_days_rest",
    "h2h_home_wins", "h2h_draws", "h2h_away_wins",
    "wc_apps_home", "wc_wins_home", "wc_apps_away", "wc_wins_away",
    "conf_home_enc", "conf_away_enc", "same_confederation",
    "is_neutral", "importance",
]


def _prep_Xy(df: pd.DataFrame, feat_cols: list[str]):
    sub = df.dropna(subset=["target"]).copy()
    available = [c for c in feat_cols if c in sub.columns]
    sub[available] = sub[available].fillna(0)
    return sub[available].values.astype(float), sub["target"].values.astype(int), available


def ece_score(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    max_probs = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (max_probs >= lo) & (max_probs <= hi if i == n_bins-1 else max_probs < hi)
        if mask.sum() == 0: continue
        ece += (mask.sum()/n) * abs(correct[mask].mean() - max_probs[mask].mean())
    return float(ece)


def train_xgboost(feat_df: pd.DataFrame, n_trials: int = 100):
    import xgboost as xgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    df = feat_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    train_mask = df["date"].dt.year <= 2020
    val_mask = df["date"].dt.year.between(2021, 2022)
    if val_mask.sum() < 50:
        val_mask = train_mask

    X_tr, y_tr, feat = _prep_Xy(df[train_mask], FEATURE_COLS)
    X_va, y_va, _    = _prep_Xy(df[val_mask],   feat)
    print(f"  XGB: train={len(X_tr)}, val={len(X_va)}, feats={len(feat)}")
    sw = np.where(y_tr == 1, 1.4, 1.0)

    def objective(trial):
        p = {
            "n_estimators":     trial.suggest_int("n_est", 200, 800),
            "max_depth":        trial.suggest_int("max_d", 3, 6),
            "learning_rate":    trial.suggest_float("lr", 0.01, 0.2, log=True),
            "subsample":        trial.suggest_float("sub", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("csbt", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("mcw", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0, 3),
            "objective": "multi:softprob", "num_class": 3,
            "verbosity": 0, "use_label_encoder": False,
        }
        m = xgb.XGBClassifier(**p)
        m.fit(X_tr, y_tr, sample_weight=sw, eval_set=[(X_va, y_va)], verbose=False)
        from sklearn.metrics import log_loss
        return log_loss(y_va, m.predict_proba(X_va))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    print(f"  XGB val_ll={study.best_value:.4f} | {best}")

    X_all, y_all, feat = _prep_Xy(df, FEATURE_COLS)
    sw_all = np.where(y_all == 1, 1.4, 1.0)
    mdl = xgb.XGBClassifier(**best, objective="multi:softprob", num_class=3,
                              verbosity=0, use_label_encoder=False)
    mdl.fit(X_all, y_all, sample_weight=sw_all)
    imp = sorted(zip(feat, mdl.feature_importances_), key=lambda x: x[1], reverse=True)
    print(f"  Top feats: {[(f, round(float(v),4)) for f,v in imp[:8]]}")
    return mdl, feat, study.best_value


def train_lgbm(feat_df: pd.DataFrame, n_trials: int = 100):
    import lightgbm as lgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    df = feat_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    train_mask = df["date"].dt.year <= 2020
    val_mask = df["date"].dt.year.between(2021, 2022)
    if val_mask.sum() < 50:
        val_mask = train_mask

    X_tr, y_tr, feat = _prep_Xy(df[train_mask], FEATURE_COLS)
    X_va, y_va, _    = _prep_Xy(df[val_mask],   feat)
    print(f"  LGBM: train={len(X_tr)}, val={len(X_va)}")
    sw = np.where(y_tr == 1, 1.4, 1.0)

    def objective(trial):
        p = {
            "n_estimators":     trial.suggest_int("n_est", 200, 800),
            "max_depth":        trial.suggest_int("max_d", 3, 7),
            "learning_rate":    trial.suggest_float("lr", 0.01, 0.2, log=True),
            "subsample":        trial.suggest_float("sub", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("csbt", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("mcw", 1, 20),
            "num_leaves":       trial.suggest_int("nl", 20, 80),
            "objective": "multiclass", "num_class": 3,
            "verbosity": -1, "force_row_wise": True,
        }
        m = lgb.LGBMClassifier(**p)
        m.fit(X_tr, y_tr, sample_weight=sw,
              eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        from sklearn.metrics import log_loss
        return log_loss(y_va, m.predict_proba(X_va))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    print(f"  LGBM val_ll={study.best_value:.4f} | {best}")

    X_all, y_all, feat = _prep_Xy(df, FEATURE_COLS)
    sw_all = np.where(y_all == 1, 1.4, 1.0)
    mdl = lgb.LGBMClassifier(**best, objective="multiclass", num_class=3,
                               verbosity=-1, force_row_wise=True)
    mdl.fit(X_all, y_all, sample_weight=sw_all)
    imp = sorted(zip(feat, mdl.feature_importances_), key=lambda x: x[1], reverse=True)
    print(f"  Top feats: {[(f, round(float(v),4)) for f,v in imp[:8]]}")
    return mdl, feat, study.best_value


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 3: FEATURE ENGINEERING + XGB + LGBM")
    print("=" * 60)

    feat_path = PROCESSED / "features.parquet"

    if feat_path.exists():
        print(f"\nLoading cached features: {feat_path}")
        feat_df = pd.read_parquet(feat_path)
        print(f"  {len(feat_df)} rows, {len(feat_df.columns)} cols")
    else:
        print("\nBuilding feature matrix from scratch ...")
        df_raw = pd.read_csv(RAW / "international_results.csv")
        df_raw.columns = df_raw.columns.str.strip()
        if "home_score" not in df_raw.columns and "home_goals" in df_raw.columns:
            df_raw = df_raw.rename(columns={"home_goals":"home_score","away_goals":"away_score"})
        df_raw["date"] = pd.to_datetime(df_raw["date"])
        t0 = time.time()
        feat_df = build_feature_matrix(df_raw, start_year=2000, min_importance=1.0)
        print(f"  Built {len(feat_df)} rows in {time.time()-t0:.1f}s")
        feat_df.to_parquet(feat_path, index=False)
        print(f"  Saved -> {feat_path}")

    if "target" in feat_df.columns:
        n = len(feat_df); c = feat_df["target"].value_counts().sort_index()
        print(f"  Class: Away={c.get(0,0)/n:.1%} Draw={c.get(1,0)/n:.1%} Home={c.get(2,0)/n:.1%}")

    print("\n[3b] Training XGBoost (100 Optuna trials) ...")
    t0 = time.time()
    xgb_mdl, xgb_feat, xgb_ll = train_xgboost(feat_df, n_trials=100)
    print(f"  XGB done in {time.time()-t0:.1f}s")
    xgb_mdl.save_model(str(MODELS / "xgb_wc2026.json"))
    with open(MODELS / "xgb_feature_cols.json","w") as f:
        json.dump({"feature_cols": xgb_feat}, f, indent=2)
    print(f"  Saved -> {MODELS / 'xgb_wc2026.json'}")

    print("\n[3c] Training LightGBM (100 Optuna trials) ...")
    t0 = time.time()
    lgbm_mdl, lgbm_feat, lgbm_ll = train_lgbm(feat_df, n_trials=100)
    print(f"  LGBM done in {time.time()-t0:.1f}s")
    lgbm_path = MODELS / "lgbm_wc2026.pkl"
    with open(lgbm_path, "wb") as f:
        pickle.dump({"model": lgbm_mdl, "feature_cols": lgbm_feat}, f)
    print(f"  Saved -> {lgbm_path}")

    print("\n=== STEP 4: CALIBRATION ===")
    feat_df2 = feat_df.copy()
    feat_df2["date"] = pd.to_datetime(feat_df2["date"])
    tour_col = feat_df2.get("tournament", pd.Series("", index=feat_df2.index))
    cal_mask = (
        feat_df2["date"].dt.year.between(2019, 2022)
        & ~tour_col.str.contains("World Cup", na=False)
    )
    if cal_mask.sum() > 50:
        X_cal, y_cal, _ = _prep_Xy(feat_df2[cal_mask], xgb_feat)
        print(f"  Calibration set: {len(X_cal)} matches")
        from sklearn.calibration import CalibratedClassifierCV
        for name, mdl, feat in [("xgb", xgb_mdl, xgb_feat), ("lgbm", lgbm_mdl, lgbm_feat)]:
            pr_before = mdl.predict_proba(X_cal)
            print(f"  {name.upper()} ECE before: {ece_score(pr_before, y_cal):.4f}")
            best_ece = 9999.0; best_cal = None; best_meth = None
            for meth in ["isotonic", "sigmoid"]:
                try:
                    cal = CalibratedClassifierCV(mdl, cv="prefit", method=meth)
                    cal.fit(X_cal, y_cal)
                    pr = cal.predict_proba(X_cal)
                    ece = ece_score(pr, y_cal)
                    print(f"  {name.upper()} ECE {meth}: {ece:.4f}")
                    if ece < best_ece:
                        best_ece = ece; best_cal = cal; best_meth = meth
                except Exception as e:
                    print(f"  {name.upper()} {meth} failed: {e}")
            if best_cal:
                out = MODELS / f"{name}_calibrated.pkl"
                with open(out, "wb") as f:
                    pickle.dump({"model": best_cal, "method": best_meth,
                                 "feature_cols": feat}, f)
                print(f"  Saved {name.upper()} calibrated ({best_meth}) -> {out}")
    else:
        print("  Insufficient calibration data")

    print("\n=== STEPS 3+4 COMPLETE ===")
