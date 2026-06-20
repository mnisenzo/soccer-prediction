"""
WC 2026 Prediction Dashboard — where your model disagrees with Kalshi.

Run:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "backtest"))

PRED_DIR = PROJECT_ROOT / "predictions"
MODELS_DIR = PROJECT_ROOT / "models"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
BACKTEST_DIR = PROJECT_ROOT / "backtest"

st.set_page_config(
    page_title="WC 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

# Kalshi uses slightly different team names than our simulation
_KALSHI_NAME_MAP: dict[str, str] = {
    "cape verde": "Cabo Verde",
    "dr congo": "Congo DR",
    "united states": "USA",
    "côte d'ivoire": "Ivory Coast",
    "cote d'ivoire": "Ivory Coast",
    "republic of ireland": "Ireland",
    "czech republic": "Czechia",
}


def _normalize_team(name: str) -> str:
    n = str(name).lower().strip()
    return _KALSHI_NAME_MAP.get(n, name)


def _to_american(p: float) -> str:
    if not isinstance(p, (int, float)) or np.isnan(p) or p <= 0 or p >= 1:
        return "N/A"
    if p > 0.5:
        return f"{-(p / (1 - p)) * 100:.0f}"
    return f"+{((1 - p) / p) * 100:.0f}"


def _pct(x) -> str:
    try:
        return f"{float(x):.1%}"
    except Exception:
        return "—"


@st.cache_data(ttl=300)
def load_sim_results() -> pd.DataFrame:
    path = PRED_DIR / "simulation_results.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_match_probs() -> pd.DataFrame:
    path = PRED_DIR / "remaining_match_probs.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=60)
def load_completed_matches() -> pd.DataFrame:
    path = RAW_DIR / "wc2026_completed.csv"
    if path.exists():
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()


@st.cache_data(ttl=60)
def load_kalshi_winner() -> pd.DataFrame:
    path = PRED_DIR / "kalshi_odds.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        winners = data.get("winner", [])
        if winners:
            return pd.DataFrame(winners)
    return pd.DataFrame()


def fetch_kalshi_live() -> pd.DataFrame:
    """Fetch Kalshi odds in-process (no subprocess)."""
    try:
        from kalshi_fetcher import get_wc_winner_odds
        df = get_wc_winner_odds()
        if not df.empty:
            PRED_DIR.mkdir(exist_ok=True)
            existing_path = PRED_DIR / "kalshi_odds.json"
            existing = {}
            if existing_path.exists():
                with open(existing_path) as f:
                    existing = json.load(f)
            existing["winner"] = df.to_dict(orient="records")
            existing["fetched_at"] = datetime.utcnow().isoformat()
            with open(existing_path, "w") as f:
                json.dump(existing, f, indent=2)
        return df
    except Exception as e:
        st.error(f"Kalshi fetch failed: {e}")
        return pd.DataFrame()


def run_simulation_subprocess(n_sims: int = 10_000) -> bool:
    # Use Anaconda Python if available (has all required packages)
    import shutil
    anaconda_py = r"C:\Users\nisen\anaconda3\python.exe"
    import os
    python_exe = anaconda_py if os.path.exists(anaconda_py) else sys.executable
    result = subprocess.run(
        [python_exe, str(PROJECT_ROOT / "src" / "simulate_tournament.py"),
         "--n-sims", str(n_sims), "--seed", "42"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        st.error(f"Simulation failed:\n{result.stderr}")
        return False
    st.cache_data.clear()
    return True


def get_standings_from_completed(completed_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute current group standings from completed matches."""
    from simulate_tournament import WC2026_GROUPS, compute_standings_from_results, sort_standings

    group_tables: dict[str, pd.DataFrame] = {}
    for group_name, teams in WC2026_GROUPS.items():
        if "group" in completed_df.columns:
            grp_df = completed_df[completed_df["group"].str.upper() == group_name.upper()]
        else:
            grp_df = pd.DataFrame()

        standings = compute_standings_from_results(teams, grp_df)
        sorted_s = sort_standings(standings)
        rows = [s.to_dict() for s in sorted_s]
        group_tables[group_name] = pd.DataFrame(rows)

    return group_tables


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚽ WC 2026 Predictor")
    st.caption("Model edge vs Kalshi prediction markets")
    st.divider()

    kalshi_data = load_kalshi_winner()
    last_fetch = "Never"
    kalshi_path = PRED_DIR / "kalshi_odds.json"
    if kalshi_path.exists():
        with open(kalshi_path) as f:
            kd = json.load(f)
        last_fetch = kd.get("fetched_at", "Unknown")[:19]

    st.caption(f"Kalshi last fetched: {last_fetch}")

    if st.button("🔄 Refresh Kalshi Odds", use_container_width=True):
        with st.spinner("Fetching live Kalshi data ..."):
            kalshi_data = fetch_kalshi_live()
        if not kalshi_data.empty:
            st.success(f"Fetched {len(kalshi_data)} contracts")
        else:
            st.warning("No Kalshi data found — markets may not be live yet")
        st.rerun()

    st.divider()

    sim_df = load_sim_results()
    sim_status = f"{len(sim_df)} teams simulated" if not sim_df.empty else "Not run yet"
    st.caption(f"Simulation: {sim_status}")

    n_sims = st.select_slider("Simulations", [1_000, 5_000, 10_000, 25_000, 50_000], value=10_000)

    if st.button("▶ Run Simulation", use_container_width=True):
        with st.spinner(f"Running {n_sims:,} Monte Carlo simulations ..."):
            ok = run_simulation_subprocess(n_sims)
        if ok:
            st.success("Simulation complete!")
            sim_df = load_sim_results()
            st.rerun()

    if st.button("↺ Refresh after new results", use_container_width=True, help="Re-run simulation using updated completed match data"):
        with st.spinner("Re-running simulation with latest results ..."):
            ok = run_simulation_subprocess(n_sims)
        if ok:
            st.success("Updated!")
            st.cache_data.clear()
            st.rerun()

    st.divider()
    page = st.radio(
        "View",
        ["Backtesting", "Upcoming Matches", "Edge Table", "Match Probabilities",
         "Group Standings", "Bracket Simulator", "Model Diagnostics"],
    )

# ─────────────────────────────────────────────────────────────
# Tab 1: Edge Table
# ─────────────────────────────────────────────────────────────

def show_edge_table():
    st.title("Where your model disagrees with Kalshi")

    sim_df = load_sim_results()
    kalshi_df = load_kalshi_winner()

    if sim_df.empty:
        st.info("Run the simulation first (sidebar → Run Simulation).")
        return

    # Merge model predictions with Kalshi odds
    model_df = sim_df[["team", "champion_pct"]].copy()
    model_df.columns = ["team", "model_prob"]

    if not kalshi_df.empty and "team" in kalshi_df.columns:
        # Normalise both sides to handle "Cape Verde" vs "Cabo Verde" etc.
        kalshi_df = kalshi_df.copy()
        kalshi_df["team_join"] = kalshi_df["team"].apply(_normalize_team)
        model_df["team_join"] = model_df["team"]
        price_cols = ["kalshi_prob", "yes_bid", "yes_ask"]
        avail_cols = [c for c in price_cols if c in kalshi_df.columns]
        merged = model_df.merge(
            kalshi_df[["team_join"] + avail_cols],
            on="team_join", how="left",
        ).drop("team_join", axis=1)
        if "yes_bid" not in merged.columns:
            merged["yes_bid"] = np.nan
        if "yes_ask" not in merged.columns:
            merged["yes_ask"] = np.nan
        merged = merged.rename(columns={"yes_bid": "kalshi_bid", "yes_ask": "kalshi_ask"})
    else:
        merged = model_df.copy()
        merged["kalshi_prob"] = np.nan
        merged["kalshi_bid"] = np.nan
        merged["kalshi_ask"] = np.nan

    merged["edge"] = merged["model_prob"] - merged["kalshi_prob"]
    merged["american_odds"] = merged["kalshi_prob"].apply(_to_american)
    merged = merged.sort_values("edge", ascending=False)

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        pos_only = st.toggle("Show only positive edge", value=False)
    with col2:
        min_edge_pct = st.slider("Min |edge| to show", 0.0, 10.0, 0.0, 0.5, format="%.1f%%")
        min_edge = min_edge_pct / 100
    with col3:
        show_all = st.checkbox("Include teams without Kalshi market", value=True)

    display = merged.copy()
    if pos_only:
        display = display[display["edge"] > 0]
    if min_edge > 0:
        display = display[display["edge"].abs() >= min_edge]
    if not show_all:
        display = display[display["kalshi_prob"].notna()]

    if display.empty:
        st.info("No teams match the current filters.")
        return

    # Build styled table
    def _color_edge(val):
        if pd.isna(val):
            return ""
        return "color: #22c55e; font-weight: bold" if val > 0 else "color: #ef4444"

    table = display[["team", "model_prob", "kalshi_prob", "edge", "american_odds"]].copy()
    table.columns = ["Team", "Your Prob", "Kalshi Prob", "Edge", "Kalshi Odds"]
    table["Your Prob"] = table["Your Prob"].apply(_pct)
    table["Kalshi Prob"] = table["Kalshi Prob"].apply(lambda x: _pct(x) if not pd.isna(x) else "—")
    table["Edge"] = table["Edge"].apply(lambda x: f"{x:+.1%}" if not pd.isna(x) else "—")

    st.dataframe(
        table,
        use_container_width=True,
        height=500,
        column_config={
            "Edge": st.column_config.TextColumn("Edge"),
        },
    )

    # Bar chart
    chart_df = merged.dropna(subset=["edge"]).head(20)
    if not chart_df.empty:
        fig = px.bar(
            chart_df.sort_values("edge"),
            x="edge", y="team",
            orientation="h",
            color="edge",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            labels={"edge": "Edge (model - market)", "team": ""},
            title="Model edge vs Kalshi (green = model higher than market)",
        )
        fig.update_layout(
            height=500,
            xaxis_tickformat=".1%",
            coloraxis_showscale=False,
            margin=dict(l=120, t=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    if kalshi_df.empty:
        st.warning("Kalshi odds not loaded. Click 'Refresh Kalshi Odds' in the sidebar.")


# ─────────────────────────────────────────────────────────────
# Tab 2: Match Probabilities
# ─────────────────────────────────────────────────────────────

def show_match_probabilities():
    st.title("Match Probabilities")

    match_df = load_match_probs()
    completed_df = load_completed_matches()

    if not completed_df.empty:
        st.subheader("Completed Matches")
        comp_display = completed_df.copy()
        comp_display["Result"] = comp_display.apply(
            lambda r: f"{r['home_team']} {int(r['home_score'])}–{int(r['away_score'])} {r['away_team']}", axis=1
        )
        comp_display["Date"] = comp_display["date"].dt.strftime("%b %d")
        st.dataframe(
            comp_display[["Date", "group", "Result", "stage"]].rename(
                columns={"group": "Group", "stage": "Stage"}
            ),
            use_container_width=True,
            hide_index=True,
        )

    if match_df.empty:
        st.info("Run simulation to see remaining match probabilities.")
        return

    st.subheader("Remaining Matches (Model Probabilities)")

    groups = sorted(match_df["group"].dropna().unique()) if "group" in match_df.columns else ["All"]
    sel_group = st.selectbox("Filter by group", ["All"] + list(groups))

    display_df = match_df if sel_group == "All" else match_df[match_df["group"] == sel_group]

    rows = []
    for _, row in display_df.iterrows():
        rows.append({
            "Group": row.get("group", ""),
            "Home": row["home_team"],
            "Away": row["away_team"],
            "Home Win": _pct(row["home_win_prob"]),
            "Draw": _pct(row["draw_prob"]),
            "Away Win": _pct(row["away_win_prob"]),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Lollipop chart for a selected match
    if not display_df.empty:
        match_labels = [f"{r['home_team']} vs {r['away_team']}" for _, r in display_df.iterrows()]
        sel_match_label = st.selectbox("Detail view", match_labels)
        idx = match_labels.index(sel_match_label)
        row = display_df.iloc[idx]

        fig = go.Figure()
        outcomes = ["Home Win", "Draw", "Away Win"]
        probs = [row["home_win_prob"], row["draw_prob"], row["away_win_prob"]]
        colors = ["#3b82f6", "#94a3b8", "#f97316"]

        for outcome, prob, color in zip(outcomes, probs, colors):
            fig.add_trace(go.Bar(
                x=[outcome], y=[prob], name=outcome,
                marker_color=color,
                text=[_pct(prob)], textposition="outside",
            ))

        fig.update_layout(
            title=f"{row['home_team']} vs {row['away_team']}",
            yaxis_tickformat=".0%", yaxis_range=[0, 1],
            showlegend=False, height=300, margin=dict(t=40),
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# Tab 3: Group Standings
# ─────────────────────────────────────────────────────────────

def show_group_standings():
    st.title("Group Standings")

    completed_df = load_completed_matches()
    sim_df = load_sim_results()

    try:
        group_tables = get_standings_from_completed(completed_df)
    except Exception as e:
        st.error(f"Could not compute standings: {e}")
        return

    groups_per_row = 3
    group_names = sorted(group_tables.keys())

    for i in range(0, len(group_names), groups_per_row):
        cols = st.columns(groups_per_row)
        for j, group_name in enumerate(group_names[i:i + groups_per_row]):
            with cols[j]:
                standings_df = group_tables[group_name]

                # Add projected champion% from simulation if available
                if not sim_df.empty and "team" in sim_df.columns:
                    standings_df = standings_df.merge(
                        sim_df[["team", "champion_pct", "advances_from_group"]],
                        on="team", how="left",
                    )
                    standings_df["Win %"] = standings_df["champion_pct"].apply(_pct)
                    standings_df["Advance %"] = standings_df["advances_from_group"].apply(_pct)

                st.markdown(f"**Group {group_name}**")
                cols_show = ["team", "played", "wins", "draws", "losses", "gf", "ga", "gd", "points"]
                if "Win %" in standings_df.columns:
                    cols_show += ["Win %", "Advance %"]
                st.dataframe(
                    standings_df[cols_show].rename(columns={"team": "Team", "played": "P",
                        "wins": "W", "draws": "D", "losses": "L",
                        "gf": "GF", "ga": "GA", "gd": "GD", "points": "Pts"}),
                    use_container_width=True,
                    hide_index=True,
                    height=200,
                )


# ─────────────────────────────────────────────────────────────
# Tab 4: Bracket Simulator
# ─────────────────────────────────────────────────────────────

def show_bracket_simulator():
    st.title("Bracket Simulator — Stage Reach Probabilities")

    sim_df = load_sim_results()
    if sim_df.empty:
        st.info("Run simulation first.")
        return

    stages = [
        ("advances_from_group", "Advances from Group"),
        ("final_pct", "Reaches Final"),
        ("champion_pct", "Wins Tournament"),
    ]

    for col, label in stages:
        if col not in sim_df.columns:
            continue
        top = sim_df[["team", col]].dropna().sort_values(col, ascending=False).head(20)
        fig = px.bar(
            top,
            x=col, y="team",
            orientation="h",
            color=col,
            color_continuous_scale="Blues",
            labels={col: "Probability", "team": ""},
            title=label,
            text=top[col].apply(_pct),
        )
        fig.update_layout(
            height=450,
            xaxis_tickformat=".0%",
            coloraxis_showscale=False,
            yaxis={"categoryorder": "total ascending"},
            margin=dict(l=130, t=40),
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap: teams × rounds
    round_cols = {
        "advances_from_group": "Groups",
        "r32_exit_pct": "R32",
        "r16_exit_pct": "R16",
        "qf_exit_pct": "QF",
        "sf_exit_pct": "SF",
        "final_pct": "Final",
        "champion_pct": "Win",
    }
    avail = {k: v for k, v in round_cols.items() if k in sim_df.columns}
    if len(avail) >= 3:
        top20 = sim_df.nlargest(20, "champion_pct")
        heat_df = top20.set_index("team")[[c for c in avail]].rename(columns=avail)

        fig = px.imshow(
            heat_df,
            color_continuous_scale="Blues",
            aspect="auto",
            title="Stage reach probability (top 20 teams by win %)",
            text_auto=".0%",
        )
        fig.update_layout(height=600, margin=dict(t=50))
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# Tab 5: Model Diagnostics
# ─────────────────────────────────────────────────────────────

def show_model_diagnostics():
    st.title("Model Diagnostics")

    st.subheader("Elo Ratings")
    elo_path = MODELS_DIR / "elo_ratings.json"
    if elo_path.exists():
        with open(elo_path) as f:
            elo = json.load(f)
        elo_df = (
            pd.DataFrame(elo.items(), columns=["team", "elo"])
            .sort_values("elo", ascending=False)
            .reset_index(drop=True)
        )
        elo_df.insert(0, "rank", range(1, len(elo_df) + 1))

        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.bar(
                elo_df.head(25), x="team", y="elo",
                color="elo", color_continuous_scale="Blues",
                labels={"elo": "Elo Rating", "team": ""},
                title="Top 25 Teams by Elo Rating",
            )
            fig.update_layout(xaxis_tickangle=-40, height=350, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.dataframe(elo_df.head(32), use_container_width=True, height=400, hide_index=True)
    else:
        st.info("Elo ratings not found. Run train_models.py first.")

    st.subheader("Dixon-Coles Attack / Defense Parameters")
    dc_path = MODELS_DIR / "dixon_coles_params.json"
    if dc_path.exists():
        with open(dc_path) as f:
            dc = json.load(f)
        teams = dc.get("teams", [])
        if teams:
            ad_df = pd.DataFrame({
                "team": teams,
                "attack": [dc["attack"].get(t, 0) for t in teams],
                "defense": [dc["defense"].get(t, 0) for t in teams],
            }).sort_values("attack", ascending=False)

            fig = px.scatter(
                ad_df, x="defense", y="attack", text="team",
                labels={"attack": "Attack strength (higher=better)", "defense": "Defense weakness (higher=worse)"},
                title="Attack vs Defense (DC model)",
                height=500,
            )
            fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
            fig.add_vline(x=0, line_dash="dash", line_color="grey", opacity=0.5)
            fig.update_traces(textposition="top center", textfont_size=9)
            st.plotly_chart(fig, use_container_width=True)

            st.caption(f"Home advantage: {dc.get('home_advantage', 0):.3f}  |  Rho: {dc.get('rho', 0):.4f}")
    else:
        st.info("Dixon-Coles params not found. Run train_models.py first.")

    st.subheader("XGBoost Feature Importance")
    xgb_feat_path = MODELS_DIR / "xgb_feature_cols.json"
    xgb_model_path = MODELS_DIR / "xgb_wc2026.json"
    if xgb_model_path.exists() and xgb_feat_path.exists():
        try:
            import xgboost as xgb
            m = xgb.XGBClassifier()
            m.load_model(str(xgb_model_path))
            with open(xgb_feat_path) as f:
                feat_cols = json.load(f)["feature_cols"]
            imp = pd.DataFrame({
                "feature": feat_cols,
                "importance": m.feature_importances_,
            }).sort_values("importance", ascending=True).tail(15)
            fig = px.bar(imp, x="importance", y="feature", orientation="h",
                         title="XGBoost Feature Importance (top 15)")
            fig.update_layout(height=400, margin=dict(l=150))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load XGBoost: {e}")
    else:
        st.info("XGBoost model not found. Run train_models.py first.")

    st.subheader("Calibration (Elo Baseline)")
    st.info("Full calibration plots require backtest computation on 2018/2022 WC — run train_models.py to generate these.")


# ─────────────────────────────────────────────────────────────
# Internet results fetch
# ─────────────────────────────────────────────────────────────

_ESPN_NAME_MAP: dict[str, str] = {
    "United States": "USA",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde": "Cabo Verde",
    "Türkiye": "Turkey",
    "Curaçao": "Curacao",
    "Ivory Coast": "Ivory Coast",
    "DR Congo": "Congo DR",
    "Congo DR": "Congo DR",
    "Côte d'Ivoire": "Ivory Coast",
    "Czech Republic": "Czechia",
    "Korea Republic": "South Korea",
}


def _espn_team(name: str) -> str:
    return _ESPN_NAME_MAP.get(name, name)


def _fetch_espn_all_dates() -> list[dict]:
    """Fetch all completed WC 2026 matches from ESPN across the tournament date range."""
    import re, requests as _req
    rows = []
    seen: set[tuple] = set()

    # Tournament date range: June 11 → July 20 2026
    from datetime import date, timedelta
    d = date(2026, 6, 11)
    end = date(2026, 7, 20)
    while d <= end:
        date_str = d.strftime("%Y%m%d")
        try:
            resp = _req.get(
                "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard",
                params={"dates": date_str}, timeout=8,
            )
            if resp.status_code != 200:
                d += timedelta(days=1)
                continue
            for ev in resp.json().get("events", []):
                comp = ev.get("competitions", [{}])[0]
                if not comp.get("status", {}).get("type", {}).get("completed"):
                    continue
                # Group from altGameNote: "FIFA World Cup, Group D"
                note = comp.get("altGameNote", "")
                m = re.search(r"Group\s+([A-L])", note)
                grp = m.group(1) if m else "?"
                # Skip if not a group stage match (note might say "Round of 32" etc)
                if "Group" not in note and grp == "?":
                    continue

                competitors = comp.get("competitors", [])
                if len(competitors) < 2:
                    continue
                home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
                away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1] if len(competitors) > 1 else {})

                home_name = _espn_team(home.get("team", {}).get("displayName", ""))
                away_name = _espn_team(away.get("team", {}).get("displayName", ""))
                home_score = int(home.get("score", 0) or 0)
                away_score = int(away.get("score", 0) or 0)
                match_date = comp.get("date", ev.get("date", ""))[:10]

                pair = (home_name, away_name)
                if pair in seen:
                    continue
                seen.add(pair)
                seen.add((away_name, home_name))

                rows.append({
                    "date": match_date,
                    "home_team": home_name,
                    "away_team": away_name,
                    "home_score": home_score,
                    "away_score": away_score,
                    "stage": "Group Stage",
                    "group": grp,
                })
        except Exception:
            pass
        d += timedelta(days=1)

    return sorted(rows, key=lambda r: (r["date"], r["group"]))


def fetch_results_from_internet(fd_api_key: str = "") -> tuple[int, str]:
    """
    Fetch all completed WC 2026 results from ESPN (no key) and update the CSV.
    Completely replaces the CSV if ESPN returns data; appends otherwise.
    Falls back to football-data.org if ESPN returns nothing.
    Returns (new_matches_added, status_message).
    """
    completed_csv = RAW_DIR / "wc2026_completed.csv"

    # Load existing to compute delta
    n_before = 0
    if completed_csv.exists():
        n_before = len(pd.read_csv(completed_csv))

    # ── ESPN (no auth, full date sweep) ───────────────────────────────────────
    new_rows = _fetch_espn_all_dates()

    if new_rows:
        df = pd.DataFrame(new_rows)
        df.to_csv(completed_csv, index=False)
        n_added = len(df) - n_before
        return max(n_added, 0), f"ESPN: {len(df)} total matches ({max(n_added,0)} new)"

    # ── football-data.org fallback ─────────────────────────────────────────────
    if fd_api_key:
        try:
            import requests as _req
            r = _req.get(
                "https://api.football-data.org/v4/competitions/WC/matches",
                params={"season": 2026, "status": "FINISHED"},
                headers={"X-Auth-Token": fd_api_key},
                timeout=12,
            )
            r.raise_for_status()
            sys.path.insert(0, str(PROJECT_ROOT / "backtest"))
            from constants import to_system_name
            fd_rows = []
            for m in r.json().get("matches", []):
                if "GROUP" not in str(m.get("stage", "")):
                    continue
                score_ft = m.get("score", {}).get("fullTime", {})
                sa, sb = score_ft.get("home"), score_ft.get("away")
                if sa is None or sb is None:
                    continue
                ta = _espn_team(to_system_name(m["homeTeam"]["name"], "fd"))
                tb = _espn_team(to_system_name(m["awayTeam"]["name"], "fd"))
                grp_raw = str(m.get("group", "")).replace("GROUP_", "")
                fd_rows.append({
                    "date": m["utcDate"][:10], "home_team": ta, "away_team": tb,
                    "home_score": int(sa), "away_score": int(sb),
                    "stage": "Group Stage", "group": grp_raw,
                })
            if fd_rows:
                df = pd.DataFrame(fd_rows)
                df.to_csv(completed_csv, index=False)
                n_added = len(df) - n_before
                return max(n_added, 0), f"football-data.org: {len(df)} total ({max(n_added,0)} new)"
        except Exception as e:
            return 0, f"football-data.org error: {e}"

    return 0, "No new results found (ESPN returned nothing; provide a football-data.org key as fallback)"


# ─────────────────────────────────────────────────────────────
# Upcoming Matches page
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=120)
def load_kalshi_match_odds() -> pd.DataFrame:
    """Load cached Kalshi match odds or fetch live."""
    cache_path = PRED_DIR / "kalshi_match_odds.json"
    if cache_path.exists():
        mtime = cache_path.stat().st_mtime
        import time
        if time.time() - mtime < 300:  # 5-min cache
            with open(cache_path) as f:
                data = json.load(f)
            return pd.DataFrame(data) if data else pd.DataFrame()
    return pd.DataFrame()


def fetch_kalshi_match_odds_live() -> pd.DataFrame:
    """Fetch live Kalshi match markets and cache."""
    try:
        from kalshi_fetcher import get_wc_match_odds
        df = get_wc_match_odds(parse=True)
        if not df.empty:
            PRED_DIR.mkdir(exist_ok=True)
            with open(PRED_DIR / "kalshi_match_odds.json", "w") as f:
                json.dump(df.to_dict(orient="records"), f)
        return df
    except Exception as e:
        st.error(f"Kalshi match fetch error: {e}")
        return pd.DataFrame()


def get_upcoming_fixtures() -> pd.DataFrame:
    """
    Return unplayed fixtures: pretournament_predictions minus completed results.
    Includes DC probs + xG columns.
    """
    preds = load_pretournament_preds()
    completed = load_completed_backtest()

    if preds.empty:
        return pd.DataFrame()

    if completed.empty:
        return preds.copy()

    # Build set of played pairs (both orderings)
    played: set[tuple] = set()
    for _, r in completed.iterrows():
        ta, tb = str(r.get("team_a", r.get("home_team", ""))), str(r.get("team_b", r.get("away_team", "")))
        played.add((ta, tb))
        played.add((tb, ta))

    mask = ~preds.apply(lambda r: (r["team_a"], r["team_b"]) in played, axis=1)
    return preds[mask].reset_index(drop=True)


def _match_kalshi_to_fixture(upcoming: pd.DataFrame, kalshi: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join upcoming fixtures with Kalshi match odds by team names.
    Handles both orderings (A vs B and B vs A).
    """
    if kalshi.empty or upcoming.empty:
        return upcoming.copy()

    result = upcoming.copy()
    for col in ["kalshi_win_a", "kalshi_draw", "kalshi_win_b", "vig"]:
        result[col] = np.nan

    for i, fix_row in result.iterrows():
        ta, tb = fix_row["team_a"], fix_row["team_b"]

        # Forward match
        fwd = kalshi[(kalshi["team_a"] == ta) & (kalshi["team_b"] == tb)]
        if not fwd.empty:
            r = fwd.iloc[0]
            result.at[i, "kalshi_win_a"] = r["kalshi_win_a"]
            result.at[i, "kalshi_draw"] = r["kalshi_draw"]
            result.at[i, "kalshi_win_b"] = r["kalshi_win_b"]
            result.at[i, "vig"] = r.get("vig", np.nan)
            continue

        # Reverse match (Kalshi lists them in opposite order)
        rev = kalshi[(kalshi["team_a"] == tb) & (kalshi["team_b"] == ta)]
        if not rev.empty:
            r = rev.iloc[0]
            result.at[i, "kalshi_win_a"] = r["kalshi_win_b"]   # flipped
            result.at[i, "kalshi_draw"] = r["kalshi_draw"]
            result.at[i, "kalshi_win_b"] = r["kalshi_win_a"]   # flipped
            result.at[i, "vig"] = r.get("vig", np.nan)

    return result


def show_upcoming_matches():
    st.title("Upcoming Matches — DC Predictions & Kalshi Odds")

    upcoming = get_upcoming_fixtures()
    if upcoming.empty:
        st.info("All fixtures have been played or pretournament predictions not generated yet.")
        if not (BACKTEST_DIR / "pretournament_predictions.csv").exists():
            st.caption("Run: python backtest/freeze_pretournament.py")
        return

    # ── Kalshi match odds fetch ───────────────────────────────────────────────
    col_k, col_info = st.columns([2, 3])
    with col_k:
        if st.button("Fetch Kalshi Match Odds", type="primary", use_container_width=True):
            with st.spinner("Fetching live Kalshi match markets ..."):
                kalshi_df = fetch_kalshi_match_odds_live()
                st.cache_data.clear()
            if not kalshi_df.empty:
                n_parsed = len(kalshi_df) if "team_a" in kalshi_df.columns else 0
                st.success(f"Fetched odds for {n_parsed} matches")
            else:
                st.warning("No match markets found — series may not be active yet")
            st.rerun()

    kalshi_df = load_kalshi_match_odds()
    with col_info:
        cache_path = PRED_DIR / "kalshi_match_odds.json"
        if cache_path.exists():
            import time
            age_min = (time.time() - cache_path.stat().st_mtime) / 60
            has_parsed = "team_a" in (kalshi_df.columns.tolist() if not kalshi_df.empty else [])
            n_k = len(kalshi_df) if has_parsed else 0
            st.caption(f"Kalshi match odds: {n_k} matches cached ({age_min:.0f} min ago)")
        else:
            st.caption("No Kalshi match odds cached yet — click Fetch above")

    # ── Merge Kalshi into upcoming ────────────────────────────────────────────
    has_kalshi = not kalshi_df.empty and "team_a" in kalshi_df.columns
    if has_kalshi:
        display = _match_kalshi_to_fixture(upcoming, kalshi_df)
    else:
        display = upcoming.copy()
        for col in ["kalshi_win_a", "kalshi_draw", "kalshi_win_b", "vig"]:
            display[col] = np.nan

    # ── Compute derived metrics ───────────────────────────────────────────────
    display["dc_exp_gd"] = display["dc_xg_a"] - display["dc_xg_b"]
    display["edge_win_a"] = display["dc_win_a"] - display["kalshi_win_a"]
    display["edge_draw"] = display["dc_draw"] - display["kalshi_draw"]
    display["edge_win_b"] = display["dc_win_b"] - display["kalshi_win_b"]

    # ── Filters ───────────────────────────────────────────────────────────────
    st.divider()
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        groups = sorted(display["group"].dropna().unique())
        sel_grp = st.selectbox("Group", ["All"] + list(groups))
    with fcol2:
        sort_by = st.selectbox("Sort by", ["Date", "DC xG Total", "Biggest Kalshi Edge", "Expected GD"])
    with fcol3:
        kalshi_only = st.toggle("Kalshi odds only", value=False,
                                help="Show only matches where Kalshi market is live")

    view = display.copy()
    if sel_grp != "All":
        view = view[view["group"] == sel_grp]
    if kalshi_only:
        view = view[view["kalshi_win_a"].notna()]

    sort_map = {
        "Date": "date",
        "DC xG Total": None,
        "Biggest Kalshi Edge": None,
        "Expected GD": "dc_exp_gd",
    }
    if sort_by == "DC xG Total":
        view = view.copy()
        view["_xg_total"] = view["dc_xg_a"] + view["dc_xg_b"]
        view = view.sort_values("_xg_total", ascending=False)
    elif sort_by == "Biggest Kalshi Edge":
        view = view.copy()
        view["_max_edge"] = view[["edge_win_a", "edge_win_b"]].abs().max(axis=1)
        view = view.sort_values("_max_edge", ascending=False)
    elif sort_by == "Expected GD":
        view = view.sort_values("dc_exp_gd", ascending=False)
    else:
        view = view.sort_values("date")

    if view.empty:
        st.info("No upcoming fixtures match the current filters.")
        return

    st.caption(f"Showing {len(view)} upcoming fixtures")

    # ── Main table ────────────────────────────────────────────────────────────
    def _fmt(v, pct=True):
        if pd.isna(v):
            return "—"
        return f"{v:.0%}" if pct else f"{v:+.0%}"

    def _fxg(v):
        return f"{v:.2f}" if pd.notna(v) else "—"

    def _fgd(v):
        return f"{v:+.2f}" if pd.notna(v) else "—"

    table_rows = []
    for _, r in view.iterrows():
        row = {
            "Grp": r["group"],
            "Date": str(r.get("date", ""))[:10],
            "Team A": r["team_a"],
            "Team B": r["team_b"],
            "DC A wins": _fmt(r["dc_win_a"]),
            "DC Draw": _fmt(r["dc_draw"]),
            "DC B wins": _fmt(r["dc_win_b"]),
            "xG A": _fxg(r.get("dc_xg_a")),
            "xG B": _fxg(r.get("dc_xg_b")),
            "Exp GD": _fgd(r.get("dc_exp_gd")),
        }
        if has_kalshi:
            row["Kal A"] = _fmt(r["kalshi_win_a"])
            row["Kal D"] = _fmt(r["kalshi_draw"])
            row["Kal B"] = _fmt(r["kalshi_win_b"])
            row["Edge A"] = _fmt(r["edge_win_a"], pct=False)
            row["Edge B"] = _fmt(r["edge_win_b"], pct=False)
            row["Vig"] = f"{r['vig']:.1%}" if pd.notna(r.get("vig")) else "—"
        table_rows.append(row)

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True, height=500)

    # ── xG scatter ───────────────────────────────────────────────────────────
    if view["dc_xg_a"].notna().any():
        st.subheader("Expected Goals — Upcoming Fixtures")
        fig_xg = go.Figure()
        for _, r in view.iterrows():
            label = f"{r['team_a']} vs {r['team_b']} (Grp {r['group']})"
            fig_xg.add_trace(go.Bar(
                name=label, x=[r["team_a"], r["team_b"]],
                y=[r["dc_xg_a"], r["dc_xg_b"]],
                showlegend=False,
                text=[f"{r['dc_xg_a']:.2f}", f"{r['dc_xg_b']:.2f}"],
                textposition="outside",
                marker_color=["#3b82f6", "#f97316"],
            ))
            break  # just show first match detail below

        # xG bubble chart: team_a xG vs team_b xG per fixture
        fig_scatter = px.scatter(
            view.dropna(subset=["dc_xg_a", "dc_xg_b"]),
            x="dc_xg_a", y="dc_xg_b",
            text=view.dropna(subset=["dc_xg_a", "dc_xg_b"]).apply(
                lambda r: f"{r['team_a'][:6]}v{r['team_b'][:6]}", axis=1
            ),
            color="group",
            size=view.dropna(subset=["dc_xg_a", "dc_xg_b"])["dc_xg_a"] + view.dropna(subset=["dc_xg_a", "dc_xg_b"])["dc_xg_b"],
            hover_data={"dc_win_a": ":.0%", "dc_draw": ":.0%", "dc_win_b": ":.0%"},
            labels={"dc_xg_a": "xG Team A", "dc_xg_b": "xG Team B", "group": "Group"},
            title="DC Expected Goals per match (size = total xG)",
        )
        fig_scatter.add_shape(type="line", x0=0, y0=0, x1=3, y1=3,
                              line=dict(dash="dash", color="grey", width=1))
        fig_scatter.add_annotation(x=2.5, y=2.7, text="Equal xG", showarrow=False,
                                   font=dict(size=10, color="grey"))
        fig_scatter.update_traces(textposition="top center", textfont_size=8)
        fig_scatter.update_layout(height=450, margin=dict(t=50))
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Kalshi edge chart ─────────────────────────────────────────────────────
    if has_kalshi and view["edge_win_a"].notna().any():
        st.subheader("Biggest Model vs Kalshi Edges (upcoming)")
        edge_data = view.dropna(subset=["edge_win_a"]).copy()
        edge_data["match"] = edge_data["team_a"] + " v " + edge_data["team_b"]
        edge_data["max_edge"] = edge_data[["edge_win_a", "edge_win_b"]].apply(
            lambda r: r["edge_win_a"] if abs(r["edge_win_a"]) > abs(r["edge_win_b"]) else r["edge_win_b"], axis=1
        )
        edge_data["edge_team"] = edge_data.apply(
            lambda r: r["team_a"] if abs(r["edge_win_a"]) > abs(r["edge_win_b"]) else r["team_b"], axis=1
        )
        top_edges = edge_data.nlargest(15, "max_edge", keep="all").sort_values("max_edge")

        fig_e = px.bar(
            top_edges, x="max_edge", y="match", orientation="h",
            color="max_edge", color_continuous_scale="RdBu", color_continuous_midpoint=0,
            text=top_edges["edge_team"],
            labels={"max_edge": "Edge (DC - Kalshi)", "match": ""},
            title="Largest DC vs Kalshi disagreement (positive = model higher)",
        )
        fig_e.update_layout(
            height=400, xaxis_tickformat=".0%",
            coloraxis_showscale=False, margin=dict(l=160, t=50),
        )
        st.plotly_chart(fig_e, use_container_width=True)

    # ── Per-match detail expanders ────────────────────────────────────────────
    st.subheader("Match Details")
    for _, r in view.iterrows():
        label = (
            f"**{r['team_a']} vs {r['team_b']}** "
            f"(Group {r['group']}, {str(r.get('date',''))[:10]})  —  "
            f"DC: {_fmt(r['dc_win_a'])} / {_fmt(r['dc_draw'])} / {_fmt(r['dc_win_b'])}  |  "
            f"xG: {_fxg(r.get('dc_xg_a'))}-{_fxg(r.get('dc_xg_b'))}"
        )
        with st.expander(label, expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(f"{r['team_a']} wins", _fmt(r["dc_win_a"]),
                          delta=_fmt(r["edge_win_a"], pct=False) if pd.notna(r.get("edge_win_a")) else None,
                          delta_color="normal")
            with c2:
                st.metric("Draw", _fmt(r["dc_draw"]))
            with c3:
                st.metric(f"{r['team_b']} wins", _fmt(r["dc_win_b"]),
                          delta=_fmt(r["edge_win_b"], pct=False) if pd.notna(r.get("edge_win_b")) else None,
                          delta_color="normal")

            xg_a = r.get("dc_xg_a")
            xg_b = r.get("dc_xg_b")
            if pd.notna(xg_a) and pd.notna(xg_b):
                st.markdown(
                    f"**DC xG:** {r['team_a']} **{xg_a:.2f}** — {r['team_b']} **{xg_b:.2f}**  |  "
                    f"Expected GD: **{xg_a - xg_b:+.2f}** (from {r['team_a']} perspective)"
                )

            if has_kalshi and pd.notna(r.get("kalshi_win_a")):
                st.markdown("**Kalshi implied (vig-removed):**")
                kc1, kc2, kc3 = st.columns(3)
                with kc1:
                    st.metric(f"{r['team_a']}", _fmt(r["kalshi_win_a"]))
                with kc2:
                    st.metric("Draw", _fmt(r["kalshi_draw"]))
                with kc3:
                    st.metric(f"{r['team_b']}", _fmt(r["kalshi_win_b"]))
                if pd.notna(r.get("vig")):
                    st.caption(f"Kalshi vig: {r['vig']:.1%}")
            elif has_kalshi:
                st.caption("No Kalshi market found for this fixture")


# ─────────────────────────────────────────────────────────────
# Tab 0: Backtesting (PRIMARY)
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=120)
def load_backtest_metrics() -> dict:
    path = BACKTEST_DIR / "metrics_2026.json"
    if path.exists():
        with open(path) as f:
            raw = json.load(f)
        # Support both flat and nested {"outcome":…,"goals":…} formats
        if "outcome" in raw or "goals" in raw:
            return raw
        # Legacy flat format — wrap as outcome only
        return {"outcome": raw}
    return {}


@st.cache_data(ttl=300)
def load_odds_comparison() -> pd.DataFrame:
    path = BACKTEST_DIR / "odds_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=120)
def load_match_rps() -> pd.DataFrame:
    path = BACKTEST_DIR / "match_rps_2026.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=120)
def load_pretournament_preds() -> pd.DataFrame:
    path = BACKTEST_DIR / "pretournament_predictions.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=120)
def load_completed_backtest() -> pd.DataFrame:
    path = BACKTEST_DIR / "completed_results.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def run_backtest_pipeline() -> bool:
    """Run the full backtest pipeline via subprocess."""
    import os
    anaconda_py = r"C:\Users\nisen\anaconda3\python.exe"
    python_exe = anaconda_py if os.path.exists(anaconda_py) else sys.executable

    steps = [
        ["backtest/get_completed_matches.py"],
        ["backtest/evaluate_2026.py"],
        ["backtest/generate_report.py"],
    ]
    for step in steps:
        result = subprocess.run(
            [python_exe] + [str(PROJECT_ROOT / s) for s in step],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            st.error(f"Step {step[0]} failed:\n{result.stderr[-1000:]}")
            return False
    st.cache_data.clear()
    return True


_WC2026_GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Switzerland", "Qatar"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USA", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cabo Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Congo DR", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}


def _get_missing_fixtures(completed_csv: Path) -> list[dict]:
    """Compare full fixture list against recorded CSV, return unplayed matches."""
    from itertools import combinations

    recorded: set[tuple] = set()
    if completed_csv.exists():
        df = pd.read_csv(completed_csv)
        for _, r in df.iterrows():
            ta, tb = str(r.get("home_team", "")), str(r.get("away_team", ""))
            grp = str(r.get("group", ""))
            recorded.add((grp, ta, tb))
            recorded.add((grp, tb, ta))

    missing = []
    for grp, teams in _WC2026_GROUPS.items():
        for ta, tb in combinations(teams, 2):
            if (grp, ta, tb) not in recorded:
                missing.append({"group": grp, "home_team": ta, "away_team": tb})
    return missing


def _append_match_to_csv(
    completed_csv: Path,
    date: str,
    group: str,
    home_team: str,
    away_team: str,
    home_score: int,
    away_score: int,
) -> None:
    """Append a single match result to the completed CSV."""
    new_row = pd.DataFrame([{
        "date": date,
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_score,
        "away_score": away_score,
        "stage": "Group Stage",
        "group": group,
    }])
    if completed_csv.exists():
        existing = pd.read_csv(completed_csv)
        updated = pd.concat([existing, new_row], ignore_index=True)
    else:
        updated = new_row
    updated.to_csv(completed_csv, index=False)


def show_match_entry():
    """Inline match entry form — shows missing fixtures and lets user record results."""
    completed_csv = RAW_DIR / "wc2026_completed.csv"
    missing = _get_missing_fixtures(completed_csv)
    n_completed = sum(len(list(__import__("itertools").combinations(t, 2))) for t in _WC2026_GROUPS.values()) - len(missing)

    st.subheader(f"Match Results ({n_completed}/72 recorded)")

    if missing:
        st.caption(f"{len(missing)} fixtures not yet recorded.")

    with st.expander("Record a match result", expanded=bool(missing)):
        # Quick-fill from missing fixtures
        if missing:
            st.markdown("**Missing fixtures** (click to prefill):")
            miss_labels = [f"Group {m['group']}: {m['home_team']} vs {m['away_team']}" for m in missing]
            quick = st.selectbox("Unrecorded match", ["(enter manually)"] + miss_labels, key="quick_fill")
        else:
            quick = "(enter manually)"
            st.success("All group fixtures recorded.")

        # Prefill values from quick-select
        prefill_group = ""
        prefill_home = ""
        prefill_away = ""
        if quick != "(enter manually)" and missing:
            idx = miss_labels.index(quick)
            m = missing[idx]
            prefill_group = m["group"]
            prefill_home = m["home_team"]
            prefill_away = m["away_team"]

        with st.form("add_match_form", clear_on_submit=True):
            col_date, col_group = st.columns([2, 1])
            with col_date:
                match_date = st.date_input("Match date", value=None, key="match_date")
            with col_group:
                group_options = list(_WC2026_GROUPS.keys())
                group_default = group_options.index(prefill_group) if prefill_group in group_options else 0
                sel_group = st.selectbox("Group", group_options, index=group_default, key="sel_group")

            teams_in_group = _WC2026_GROUPS[sel_group]
            home_default = teams_in_group.index(prefill_home) if prefill_home in teams_in_group else 0
            away_options = [t for t in teams_in_group if t != teams_in_group[home_default]]
            if prefill_away in away_options:
                away_default = away_options.index(prefill_away)
            else:
                away_default = 0

            col_home, col_score, col_away = st.columns([3, 1, 3])
            with col_home:
                home_team = st.selectbox("Home team", teams_in_group, index=home_default, key="home_team")
            with col_score:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**vs**")
            with col_away:
                away_opts = [t for t in teams_in_group if t != home_team]
                away_team = st.selectbox("Away team", away_opts,
                                         index=min(away_default, len(away_opts)-1), key="away_team")

            col_hs, col_spacer, col_as = st.columns([2, 3, 2])
            with col_hs:
                home_score = st.number_input(f"{home_team} goals", min_value=0, max_value=30, value=0, step=1)
            with col_as:
                away_score = st.number_input(f"{away_team} goals", min_value=0, max_value=30, value=0, step=1)

            submitted = st.form_submit_button("Save result & re-evaluate", type="primary", use_container_width=True)

        if submitted:
            if match_date is None:
                st.error("Please select a match date.")
            else:
                # Check for duplicate
                dup = False
                if completed_csv.exists():
                    existing = pd.read_csv(completed_csv)
                    dup = (
                        ((existing["home_team"] == home_team) & (existing["away_team"] == away_team)) |
                        ((existing["home_team"] == away_team) & (existing["away_team"] == home_team))
                    ).any()
                if dup:
                    st.warning(f"{home_team} vs {away_team} is already recorded. Remove it from the CSV first to update.")
                else:
                    _append_match_to_csv(
                        completed_csv,
                        date=str(match_date),
                        group=sel_group,
                        home_team=home_team,
                        away_team=away_team,
                        home_score=int(home_score),
                        away_score=int(away_score),
                    )
                    st.success(f"Saved: {home_team} {int(home_score)}-{int(away_score)} {away_team}")
                    st.cache_data.clear()

                    # Auto re-evaluate
                    with st.spinner("Re-evaluating ..."):
                        ok = run_backtest_pipeline()
                    if ok:
                        st.success("Evaluation updated.")
                    st.rerun()

    # Show current recorded matches with option to delete
    if completed_csv.exists():
        df = pd.read_csv(completed_csv).sort_values("date")
        df["Result"] = df.apply(
            lambda r: f"{r['home_team']} {int(r['home_score'])}-{int(r['away_score'])} {r['away_team']}", axis=1
        )
        st.dataframe(
            df[["date", "group", "Result"]].rename(columns={"date": "Date", "group": "Group"}),
            use_container_width=True,
            hide_index=True,
            height=min(400, 40 + len(df) * 35),
        )


def _edge_color(edge: float) -> str:
    if pd.isna(edge):
        return "color: grey"
    if edge > 0.02:
        return "color: #15803d; font-weight: bold"
    if edge > 0.005:
        return "color: #22c55e"
    if edge < -0.02:
        return "color: #b91c1c; font-weight: bold"
    if edge < -0.005:
        return "color: #ef4444"
    return "color: grey"


def _bt_odds_tab(odds_df: pd.DataFrame) -> None:
    st.subheader("Championship Odds vs Model")

    if odds_df.empty:
        st.info(
            "No odds comparison data. "
            "Run `python backtest/odds_comparison.py` to generate it."
        )
        return

    overround = odds_df["raw_implied_prob"].sum()
    st.caption(
        f"Market probabilities derived from American pre-tournament odds with vig removed. "
        f"Overround: {overround:.3f} ({(overround-1)*100:.1f}% vig). Odds sourced pre-tournament."
    )

    edge_only = st.toggle("Show only |edge| > 1%", value=False)

    # Keep numeric cols as floats so column sorting works correctly.
    # Use Styler.format() for percentage display only.
    display = odds_df.rename(columns={
        "team": "Team", "american_odds": "Odds",
        "model_prob": "Model %", "fair_implied_prob": "Market %",
        "edge": "Edge",
    })[["Team", "Model %", "Market %", "Edge", "Odds"]].copy()

    if edge_only:
        display = display[display["Edge"].abs() > 0.01]

    def _color_edge_col(s):
        if s.name != "Edge":
            return [""] * len(s)
        return [_edge_color(v) for v in s]

    st.dataframe(
        display.style
            .format({"Model %": lambda v: f"{v:.2%}" if pd.notna(v) else "—",
                     "Market %": "{:.2%}",
                     "Edge": lambda v: f"{v:+.2%}" if pd.notna(v) else "—"})
            .apply(_color_edge_col),
        use_container_width=True,
        hide_index=True,
        height=500,
    )

    # Bar chart — top 20 model teams
    top20 = odds_df.dropna(subset=["model_prob"]).nlargest(20, "model_prob")
    bar_df = pd.DataFrame({
        "Team":   list(top20["team"]) * 2,
        "Prob":   list(top20["model_prob"]) + list(top20["fair_implied_prob"]),
        "Source": ["Model"] * len(top20) + ["Market"] * len(top20),
    })
    fig = px.bar(
        bar_df, x="Team", y="Prob", color="Source", barmode="group",
        color_discrete_map={"Model": "#3b82f6", "Market": "#f59e0b"},
        title="Championship Probability: Model vs Market (top 20 by model)",
        labels={"Prob": "Probability"},
    )
    fig.update_layout(height=420, xaxis_tickangle=-35, yaxis_tickformat=".1%",
                      margin=dict(b=120, t=50))
    st.plotly_chart(fig, use_container_width=True)


def _bt_outcome_tab(metrics: dict, match_df: pd.DataFrame, preds: pd.DataFrame) -> None:
    outcome = metrics.get("outcome", metrics)  # support old flat format
    n_matches = outcome.get("ens", outcome.get("dc", {})).get("n", 0)
    st.caption(f"**{n_matches} matches evaluated.** Bootstrap CIs are wide at this sample size.")

    # Metrics table
    st.subheader("Model Performance Summary")
    model_order  = ["ens", "dc", "xgb", "lgbm", "elo", "baseline_uniform"]
    model_labels = {
        "ens": "Ensemble", "dc": "Dixon-Coles", "xgb": "XGBoost",
        "lgbm": "LightGBM", "elo": "Elo", "baseline_uniform": "Baseline (uniform 1/3)",
    }
    rows = []
    for k in model_order:
        if k not in outcome:
            continue
        m   = outcome[k]
        ci  = f"[{m['rps_ci_lo']:.3f}, {m['rps_ci_hi']:.3f}]"
        ece = m.get("ece_confidence")
        rows.append({
            "Model":     model_labels.get(k, k),
            "RPS":       round(m["rps_mean"], 4),
            "95% CI":    ci,
            "Log-Loss":  round(m["log_loss"], 4),
            "Accuracy":  f"{m['accuracy']:.1%}",
            "ECE":       f"{ece:.4f}" if ece is not None else "—",
        })
    if rows:
        mdf = pd.DataFrame(rows)
        def _hl(s):
            if s.name != "RPS": return [""] * len(s)
            mn = s.min()
            return ["background-color:#bbf7d0;font-weight:bold" if v == mn else "" for v in s]
        st.dataframe(mdf.style.apply(_hl), use_container_width=True, hide_index=True)

    # Bootstrap note
    if "ens" in outcome and "baseline_uniform" in outcome:
        ens_rps  = np.array(outcome["ens"].get("rps_per_match", []))
        base_rps = np.array(outcome["baseline_uniform"].get("rps_per_match", []))
        if len(ens_rps) == len(base_rps) > 0:
            diff = ens_rps - base_rps
            from numpy import percentile as _pct
            ci_lo = float(_pct(np.random.default_rng(42).choice(diff, (10000, len(diff)), replace=True).mean(axis=1), 2.5))
            ci_hi = float(_pct(np.random.default_rng(42).choice(diff, (10000, len(diff)), replace=True).mean(axis=1), 97.5))
            msg = (
                f"Ensemble vs Uniform bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]  "
                f"(avg diff {diff.mean():+.4f})"
            )
            if ci_lo <= 0 <= ci_hi:
                st.warning(f"{msg}\nDifference not yet statistically significant at n={n_matches}.")
            else:
                st.info(msg)

    # Reliability diagram (calibration curve)
    cal_bins = outcome.get("ens", {}).get("calibration_bins", [])
    if cal_bins:
        st.subheader("Reliability Diagram (Ensemble)")
        st.caption(
            "Each dot = one confidence bucket. Perfectly calibrated predictions follow the diagonal. "
            "Points above the line = underconfident; below = overconfident. "
            f"ECE = {outcome['ens'].get('ece_confidence', float('nan')):.4f}"
        )
        cal_df = pd.DataFrame(cal_bins)
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color="grey", width=1),
            name="Perfect calibration",
        ))
        fig_cal.add_trace(go.Scatter(
            x=cal_df["confidence"], y=cal_df["accuracy"],
            mode="markers+lines",
            marker=dict(size=cal_df["n"].clip(upper=100) / 10 + 5, color="#3b82f6"),
            text=[f"n={r['n']}<br>conf={r['confidence']:.3f}<br>acc={r['accuracy']:.3f}" for _, r in cal_df.iterrows()],
            hoverinfo="text",
            name="Ensemble",
        ))
        fig_cal.update_layout(
            height=350, xaxis_title="Mean predicted confidence", yaxis_title="Fraction correct",
            xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
            margin=dict(t=20, b=40), legend=dict(x=0.02, y=0.95),
        )
        st.plotly_chart(fig_cal, use_container_width=True)

    # RPS bar chart
    if not match_df.empty:
        st.subheader("Per-Match RPS (Ensemble)")
        st.caption("Lower = better. Red = model predicted wrong outcome.")
        sdf = match_df.sort_values("date").reset_index(drop=True)
        labels = [f"{r['team_a'][:7]}v{r['team_b'][:7]} ({r['group']})" for _, r in sdf.iterrows()]
        colors = ["#ef4444" if u else "#3b82f6" for u in sdf["is_upset"]]
        fig = go.Figure(go.Bar(
            x=labels, y=sdf["rps"].values,
            marker_color=colors,
            text=[f"{v:.3f}" for v in sdf["rps"].values],
            textposition="outside",
        ))
        fig.add_hline(y=float(sdf["rps"].mean()), line_dash="dash", line_color="grey",
                      annotation_text=f"avg {sdf['rps'].mean():.3f}")
        fig.update_layout(height=340, xaxis_tickangle=-40, yaxis_title="RPS",
                          yaxis_range=[0, sdf["rps"].max() * 1.25], margin=dict(t=30, b=120))
        st.plotly_chart(fig, use_container_width=True)

    # Cumulative RPS
    if not match_df.empty:
        sdf = match_df.sort_values("date").reset_index(drop=True)
        sdf["cum_rps"] = sdf["rps"].expanding().mean()
        sdf["cum_acc"] = (~sdf["is_upset"]).cumsum() / (sdf.index + 1)
        sdf["lbl"] = [f"{r['team_a'][:5]}v{r['team_b'][:5]}" for _, r in sdf.iterrows()]
        t1, t2 = st.tabs(["Cumulative RPS", "Cumulative Accuracy"])
        with t1:
            fig = px.line(sdf, x="lbl", y="cum_rps", markers=True,
                          labels={"lbl": "", "cum_rps": "Cumul. avg RPS"},
                          title="Cumulative Average RPS (lower is better)")
            fig.add_hline(y=1/3, line_dash="dot", line_color="orange", annotation_text="uniform (0.333)")
            fig.update_layout(height=300, xaxis_tickangle=-30, margin=dict(b=80))
            st.plotly_chart(fig, use_container_width=True)
        with t2:
            fig = px.line(sdf, x="lbl", y="cum_acc", markers=True,
                          labels={"lbl": "", "cum_acc": "Accuracy"},
                          title="Cumulative Accuracy")
            fig.update_layout(height=300, xaxis_tickangle=-30, yaxis_tickformat=".0%",
                              yaxis_range=[0, 1], margin=dict(b=80))
            st.plotly_chart(fig, use_container_width=True)

    # Match-by-match expanders
    if not match_df.empty:
        st.subheader("Match-by-Match Breakdown")
        for _, u in match_df.sort_values("date").iterrows():
            ta, tb   = str(u["team_a"]), str(u["team_b"])
            grp      = str(u.get("group", "?"))
            date_s   = str(u.get("date", ""))[:10]
            score    = str(u.get("score", "?-?"))
            is_upset = bool(u.get("is_upset", False))
            rps_val  = float(u.get("rps", 0))
            p_act    = float(u.get("prob_of_actual", 0))

            label = (
                f"{'❌' if is_upset else '✅'} **{ta} vs {tb}** ({grp}, {date_s}) — "
                f"{score} → {u.get('outcome_label','?')} | RPS {rps_val:.3f} | p={p_act:.0%}"
            )
            with st.expander(label, expanded=False):
                c1, c2, c3 = st.columns(3)
                c1.metric(f"{ta} wins", f"{float(u.get('ens_prob_a',0)):.0%}")
                c2.metric("Draw",        f"{float(u.get('ens_draw',0)):.0%}")
                c3.metric(f"{tb} wins", f"{float(u.get('ens_prob_b',0)):.0%}")

                if not preds.empty:
                    mp = preds[
                        ((preds["team_a"]==ta)&(preds["team_b"]==tb)) |
                        ((preds["team_a"]==tb)&(preds["team_b"]==ta))
                    ]
                    if not mp.empty:
                        row     = mp.iloc[0]
                        flipped = row["team_a"] != ta
                        sfx_a, sfx_b = ("b","a") if flipped else ("a","b")
                        mrows = []
                        for mn in ["dc","elo","ens"]:
                            ca, cd, cb = f"{mn}_win_{sfx_a}", f"{mn}_draw", f"{mn}_win_{sfx_b}"
                            if ca in row.index and pd.notna(row[ca]):
                                mrows.append({"Model": mn.upper(),
                                              f"{ta} wins": f"{float(row[ca]):.0%}",
                                              "Draw": f"{float(row[cd]):.0%}",
                                              f"{tb} wins": f"{float(row[cb]):.0%}"})
                        if mrows:
                            st.dataframe(pd.DataFrame(mrows), use_container_width=True, hide_index=True)
                        if "dc_xg_a" in row.index and pd.notna(row.get("dc_xg_a")):
                            xa = float(row["dc_xg_a"] if not flipped else row["dc_xg_b"])
                            xb = float(row["dc_xg_b"] if not flipped else row["dc_xg_a"])
                            try:
                                sa_, sb_ = [int(x) for x in score.split("-")]
                            except Exception:
                                sa_, sb_ = 0, 0
                            st.caption(
                                f"DC xG: {ta} {xa:.2f} — {tb} {xb:.2f}  |  "
                                f"Actual: {ta} {sa_} — {tb} {sb_}"
                            )
                            if "dc_pred_score_a" in row.index and pd.notna(row.get("dc_pred_score_a")):
                                pa_ = int(row["dc_pred_score_a"] if not flipped else row["dc_pred_score_b"])
                                pb_ = int(row["dc_pred_score_b"] if not flipped else row["dc_pred_score_a"])
                                st.caption(f"Modal predicted scoreline: {ta} {pa_} — {tb} {pb_}")


def _bt_goals_tab(metrics: dict, preds: pd.DataFrame, completed: pd.DataFrame) -> None:
    st.subheader("Goals Prediction (Dixon-Coles)")
    goals = metrics.get("goals", {})

    if not goals:
        st.info("Goals metrics not yet computed. Re-run evaluation pipeline.")
        return

    g = goals
    st.subheader("Summary Metrics")
    gdf = pd.DataFrame([
        {"Metric": "MAE — Team A xG",        "Value": f"{g['mae_a']:.3f}",     "Baseline": f"{1.35:.3f}"},
        {"Metric": "MAE — Team B xG",        "Value": f"{g['mae_b']:.3f}",     "Baseline": f"{1.35:.3f}"},
        {"Metric": "MAE — Average",           "Value": f"{g['mae_avg']:.3f}",   "Baseline": f"{1.35:.3f}"},
        {"Metric": "RMSE — Team A",           "Value": f"{g['rmse_a']:.3f}",    "Baseline": "—"},
        {"Metric": "RMSE — Team B",           "Value": f"{g['rmse_b']:.3f}",    "Baseline": "—"},
        {"Metric": "Total goals MAE",         "Value": f"{g['total_goals_mae']:.3f}", "Baseline": "—"},
        {"Metric": "Goal diff MAE",           "Value": f"{g['goal_diff_mae']:.3f}",  "Baseline": "—"},
        {"Metric": "Directional acc (GD)",    "Value": f"{g['directional_accuracy']:.1%}", "Baseline": "—"},
        {"Metric": "Exact scoreline acc",     "Value": f"{g['exact_score_acc']:.1%}",  "Baseline": "—"},
        {"Metric": "±1 goal both sides",      "Value": f"{g['within1_acc']:.1%}",      "Baseline": "—"},
        {"Metric": "Mean scoreline log-P",    "Value": f"{g['scoreline_log_p']:.4f}",  "Baseline": f"{g['naive_log_p']:.4f}"},
        {"Metric": "Bias team_a (+ = over)", "Value": f"{g['bias_a']:+.3f}",    "Baseline": "0"},
        {"Metric": "Bias team_b (+ = over)", "Value": f"{g['bias_b']:+.3f}",    "Baseline": "0"},
        {"Metric": "Bias strong teams",      "Value": f"{g['bias_strong']:+.3f}" if pd.notna(g.get('bias_strong')) else "—", "Baseline": "0"},
        {"Metric": "Bias weak teams",        "Value": f"{g['bias_weak']:+.3f}"  if pd.notna(g.get('bias_weak'))   else "—", "Baseline": "0"},
    ])
    st.dataframe(gdf, use_container_width=True, hide_index=True)

    # Scatter plots
    if not preds.empty and not completed.empty:
        needed = ["dc_xg_a", "dc_xg_b", "dc_pred_score_a", "dc_pred_score_b"]
        if all(c in preds.columns for c in needed):
            from backtest.evaluate_2026 import merge_predictions_results
            try:
                merged = merge_predictions_results(preds, completed)
                scatter_df = merged.dropna(subset=["dc_xg_a","dc_xg_b","score_a","score_b"])
                if not scatter_df.empty:
                    scatter_df = scatter_df.copy()
                    scatter_df["match"] = scatter_df["team_a"] + " v " + scatter_df["team_b"]
                    scatter_df["err_a"] = (scatter_df["dc_xg_a"] - scatter_df["score_a"]).round(2)
                    scatter_df["err_b"] = (scatter_df["dc_xg_b"] - scatter_df["score_b"]).round(2)

                    col1, col2 = st.columns(2)
                    axis_max = max(
                        scatter_df[["dc_xg_a","dc_xg_b","score_a","score_b"]].max().max() + 0.5, 4.0
                    )
                    with col1:
                        fig = px.scatter(
                            scatter_df, x="dc_xg_a", y="score_a",
                            hover_name="match",
                            hover_data={"dc_xg_a": ":.2f", "score_a": True, "err_a": ":.2f"},
                            labels={"dc_xg_a": "Predicted xG", "score_a": "Actual Goals"},
                            title="Team A: xG vs Actual",
                            color="group",
                        )
                        fig.add_shape(type="line", x0=0, y0=0, x1=axis_max, y1=axis_max,
                                      line=dict(dash="dot", color="grey"))
                        fig.update_layout(height=360, xaxis_range=[0, axis_max], yaxis_range=[0, axis_max])
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        fig = px.scatter(
                            scatter_df, x="dc_xg_b", y="score_b",
                            hover_name="match",
                            hover_data={"dc_xg_b": ":.2f", "score_b": True, "err_b": ":.2f"},
                            labels={"dc_xg_b": "Predicted xG", "score_b": "Actual Goals"},
                            title="Team B: xG vs Actual",
                            color="group",
                        )
                        fig.add_shape(type="line", x0=0, y0=0, x1=axis_max, y1=axis_max,
                                      line=dict(dash="dot", color="grey"))
                        fig.update_layout(height=360, xaxis_range=[0, axis_max], yaxis_range=[0, axis_max])
                        st.plotly_chart(fig, use_container_width=True)

                    # Bias bar
                    strong = {"France","Spain","England","Argentina","Portugal","Brazil",
                              "Germany","Netherlands","Norway","Morocco","USA","Belgium",
                              "Colombia","Japan","Mexico"}
                    bias_data = {"Category": ["Team A overall", "Team B overall",
                                               "Strong teams (A)", "Weak teams (A)"],
                                 "Bias": [g["bias_a"], g["bias_b"],
                                           g.get("bias_strong", 0) or 0,
                                           g.get("bias_weak", 0) or 0]}
                    bdf = pd.DataFrame(bias_data)
                    bdf["color"] = bdf["Bias"].apply(lambda x: "#ef4444" if x > 0 else "#3b82f6")
                    fig = go.Figure(go.Bar(
                        x=bdf["Category"], y=bdf["Bias"],
                        marker_color=bdf["color"],
                        text=[f"{v:+.3f}" for v in bdf["Bias"]],
                        textposition="outside",
                    ))
                    fig.add_hline(y=0, line_color="black")
                    fig.update_layout(height=300, yaxis_title="Bias (+ = over-predicted)",
                                      title="Systematic xG Bias (red = overpredicted, blue = underpredicted)",
                                      margin=dict(t=50))
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render scatter plots: {e}")


def show_backtesting():
    st.title("Backtesting — WC 2026 Group Stage")
    st.caption("Pre-tournament predictions frozen before June 11, 2026. Evaluated against actual results.")

    # ── Controls ──────────────────────────────────────────────────────────────
    with st.expander("Fetch latest results from internet", expanded=False):
        fd_key = st.text_input("football-data.org API key (optional)", value="",
                               type="password", key="fd_api_key")
        if st.button("Fetch & Update Results", use_container_width=True):
            with st.spinner("Fetching ..."):
                n_new, msg = fetch_results_from_internet(fd_api_key=fd_key)
            if n_new > 0:
                st.success(f"{msg} — re-running evaluation ...")
                st.cache_data.clear()
                with st.spinner("Re-evaluating ..."):
                    run_backtest_pipeline()
                st.cache_data.clear()
                st.rerun()
            else:
                st.info(msg)

    show_match_entry()
    st.divider()

    col_run, col_report, col_odds = st.columns([2, 1, 1])
    with col_run:
        if st.button("Re-run Evaluation", type="primary", use_container_width=True):
            with st.spinner("Running evaluation pipeline..."):
                ok = run_backtest_pipeline()
            if ok:
                st.success("Backtest complete!")
            st.cache_data.clear()
            st.rerun()

    with col_report:
        report_path = BACKTEST_DIR / "report_2026.md"
        if report_path.exists():
            with open(report_path, encoding="utf-8") as f:
                st.download_button("Download Report", data=f.read(),
                                   file_name="wc2026_backtest_report.md",
                                   mime="text/markdown", use_container_width=True)

    with col_odds:
        if st.button("Refresh Odds Comparison", use_container_width=True):
            import os
            py = r"C:\Users\nisen\anaconda3\python.exe"
            if not os.path.exists(py): py = sys.executable
            with st.spinner("Running odds comparison ..."):
                res = subprocess.run([py, str(PROJECT_ROOT / "backtest/odds_comparison.py")],
                                     capture_output=True, text=True, cwd=str(PROJECT_ROOT))
            if res.returncode == 0:
                st.success("Done"); st.cache_data.clear(); st.rerun()
            else:
                st.error(res.stderr[-300:])

    # ── Load data ────────────────────────────────────────────────────────────
    metrics   = load_backtest_metrics()
    match_df  = load_match_rps()
    preds     = load_pretournament_preds()
    completed = load_completed_backtest()
    odds_df   = load_odds_comparison()

    if not metrics and match_df.empty:
        st.info("No backtest data yet. Click **Re-run Evaluation** above.")
        if not completed.empty:
            st.subheader(f"Completed Matches ({len(completed)})")
            st.dataframe(completed[["date","group","team_a","team_b","score_a","score_b","outcome"]],
                         use_container_width=True, hide_index=True)
        return

    # ── Three-tab layout ─────────────────────────────────────────────────────
    tab_odds, tab_outcome, tab_goals = st.tabs([
        "Championship Odds vs Model",
        "Match Outcome Predictions",
        "Goals Predictions",
    ])

    with tab_odds:
        _bt_odds_tab(odds_df)

    with tab_outcome:
        _bt_outcome_tab(metrics, match_df, preds)

    with tab_goals:
        _bt_goals_tab(metrics, preds, completed)


# ─────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────

if page == "Backtesting":
    show_backtesting()
elif page == "Upcoming Matches":
    show_upcoming_matches()
elif page == "Edge Table":
    show_edge_table()
elif page == "Match Probabilities":
    show_match_probabilities()
elif page == "Group Standings":
    show_group_standings()
elif page == "Bracket Simulator":
    show_bracket_simulator()
elif page == "Model Diagnostics":
    show_model_diagnostics()
