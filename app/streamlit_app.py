"""Streamlit frontend for the soccer prediction system."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make the src package importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from soccer_prediction.data.database import init_db, get_session
from soccer_prediction.data.loaders import DataLoader
from soccer_prediction.data.repositories import (
    get_completed_matches,
    get_fixtures,
    get_group_assignments,
    get_teams,
)
from soccer_prediction.markets.evaluator import MarketEvaluator
from soccer_prediction.markets.registry import MarketRegistry
from soccer_prediction.models.registry import ModelRegistry, list_models
from soccer_prediction.simulation.monte_carlo import MonteCarloSimulator, SimulationResults
from soccer_prediction.simulation.tournament import DEFAULT_WC2026_GROUPS, TournamentConfig

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="World Cup Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).parent.parent
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
MARKETS_DIR = PROJECT_ROOT / "configs" / "markets"


# ─── Bootstrap ────────────────────────────────────────────────────────────────

@st.cache_resource
def _init():
    """One-time DB initialisation."""
    init_db()


_init()


# ─── Shared state helpers ──────────────────────────────────────────────────────

def _session_default(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


_session_default("sim_results", None)
_session_default("model_registry", ModelRegistry())
_session_default("market_registry", MarketRegistry())
_session_default("model_trained", False)


@st.cache_data(show_spinner=False)
def load_teams_df() -> pd.DataFrame:
    with get_session() as s:
        return get_teams(s)


@st.cache_data(show_spinner=False)
def load_matches_df() -> pd.DataFrame:
    with get_session() as s:
        return get_completed_matches(s)


@st.cache_data(show_spinner=False)
def load_fixtures_df() -> pd.DataFrame:
    with get_session() as s:
        return get_fixtures(s, "FIFA World Cup", 2026)


@st.cache_data(show_spinner=False)
def load_group_assignments() -> dict:
    with get_session() as s:
        return get_group_assignments(s, "FIFA World Cup", 2026) or DEFAULT_WC2026_GROUPS


def invalidate_data_caches():
    load_teams_df.clear()
    load_matches_df.clear()
    load_fixtures_df.clear()
    load_group_assignments.clear()


# ─── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("⚽ WC 2026 Predictor")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Match Predictor", "Tournament Simulation", "Market Analysis",
     "Feature Inspector", "Data Manager"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.caption("Model")
selected_model = st.sidebar.selectbox("Active model", list_models(), index=0, key="selected_model")

st.sidebar.divider()
st.sidebar.caption("Quick actions")
if st.sidebar.button("Load sample data"):
    loader = DataLoader()
    teams_csv = SAMPLE_DIR / "teams.csv"
    hist_csv = SAMPLE_DIR / "historical_matches.csv"
    fixtures_csv = SAMPLE_DIR / "wc_2026_fixtures.csv"
    msgs = []
    if teams_csv.exists():
        n = loader.load_teams(teams_csv)
        msgs.append(f"{n} teams")
    if hist_csv.exists():
        n = loader.load_historical_matches(hist_csv)
        msgs.append(f"{n} historical matches")
    if fixtures_csv.exists():
        n = loader.load_fixtures(fixtures_csv, "FIFA World Cup", 2026)
        msgs.append(f"{n} fixtures")
    invalidate_data_caches()
    st.sidebar.success("Loaded: " + ", ".join(msgs) if msgs else "No sample files found.")

if st.sidebar.button("Train model"):
    matches = load_matches_df()
    if matches.empty:
        st.sidebar.error("No historical data loaded.")
    else:
        reg: ModelRegistry = st.session_state["model_registry"]
        model = reg.fit_and_register(selected_model, matches)
        st.session_state["model_trained"] = True
        st.sidebar.success(f"Model '{model.name}' trained on {len(matches)} matches.")

st.sidebar.divider()
if st.session_state["model_trained"]:
    st.sidebar.success(f"Model trained: {selected_model}")
else:
    st.sidebar.warning("No model trained yet.")


# ─── Overview ─────────────────────────────────────────────────────────────────

def show_overview():
    st.title("World Cup 2026 Prediction System")
    st.markdown(
        "A modular prediction framework for FIFA World Cup outcomes and "
        "Kalshi-style prediction markets."
    )

    teams_df = load_teams_df()
    matches_df = load_matches_df()
    fixtures_df = load_fixtures_df()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Teams in DB", len(teams_df))
    c2.metric("Historical matches", len(matches_df))
    c3.metric("WC 2026 fixtures", len(fixtures_df))
    mkt_reg: MarketRegistry = st.session_state["market_registry"]
    if len(mkt_reg) == 0:
        mkt_reg.load_directory(MARKETS_DIR)
    c4.metric("Markets loaded", len(mkt_reg))

    if not teams_df.empty:
        st.subheader("Top 20 Teams by Elo Rating")
        fig = px.bar(
            teams_df.head(20),
            x="name", y="elo_rating", color="confederation",
            labels={"name": "Team", "elo_rating": "Elo Rating"},
            height=350,
        )
        fig.update_layout(xaxis_tickangle=-40, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Quick Start")
    st.markdown("""
1. **Load sample data** (sidebar) → populates teams, historical matches, WC 2026 fixtures
   *If `historical_matches.csv` is missing, run: `python scripts/generate_sample_data.py`*
2. **Train model** (sidebar) → fits the selected model on historical data
3. Go to **Match Predictor** to predict individual matches
4. Go to **Tournament Simulation** to run Monte Carlo and see advancement probabilities
5. Go to **Market Analysis** to compare model probabilities to Kalshi prices
    """)


# ─── Match Predictor ──────────────────────────────────────────────────────────

def show_match_predictor():
    st.title("Match Predictor")

    teams_df = load_teams_df()
    if teams_df.empty:
        st.warning("No teams loaded. Use 'Load sample data' in the sidebar.")
        return

    reg: ModelRegistry = st.session_state["model_registry"]
    model = reg.get(selected_model)
    if model is None:
        st.warning("No model trained. Click 'Train model' in the sidebar first.")
        return

    team_names = sorted(teams_df["name"].tolist())

    col1, col2, col3 = st.columns([3, 1, 3])
    with col1:
        home = st.selectbox("Home / Team A", team_names, index=team_names.index("USA") if "USA" in team_names else 0)
    with col2:
        st.markdown("<br><br><center>vs</center>", unsafe_allow_html=True)
    with col3:
        away_default = team_names.index("Germany") if "Germany" in team_names else 1
        away = st.selectbox("Away / Team B", team_names, index=away_default)

    is_neutral = st.checkbox("Neutral venue", value=True)

    if st.button("Predict", type="primary"):
        pred = model.predict_match(home, away, is_neutral=is_neutral)

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{home} Win", f"{pred.home_win_prob:.1%}")
        c2.metric("Draw", f"{pred.draw_prob:.1%}")
        c3.metric(f"{away} Win", f"{pred.away_win_prob:.1%}")

        # Gauge chart
        fig = go.Figure(go.Bar(
            x=[pred.home_win_prob, pred.draw_prob, pred.away_win_prob],
            y=[f"{home} Win", "Draw", f"{away} Win"],
            orientation="h",
            text=[f"{p:.1%}" for p in [pred.home_win_prob, pred.draw_prob, pred.away_win_prob]],
            textposition="inside",
            marker_color=["#2196F3", "#9E9E9E", "#F44336"],
        ))
        fig.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        if pred.home_goals_exp is not None:
            st.subheader("Expected Goals")
            col_a, col_b = st.columns(2)
            col_a.metric(f"xG {home}", f"{pred.home_goals_exp:.2f}")
            col_b.metric(f"xG {away}", f"{pred.away_goals_exp:.2f}")

        if pred.score_matrix is not None:
            st.subheader("Score Probability Heatmap")
            max_g = 6
            mat = pred.score_matrix[:max_g + 1, :max_g + 1]
            fig_h = px.imshow(
                mat * 100,
                labels={"x": f"{away} Goals", "y": f"{home} Goals", "color": "Prob %"},
                x=[str(i) for i in range(max_g + 1)],
                y=[str(i) for i in range(max_g + 1)],
                color_continuous_scale="Blues",
                text_auto=".1f",
            )
            fig_h.update_layout(height=400, margin=dict(t=10))
            st.plotly_chart(fig_h, use_container_width=True)

        # Market-like stats
        st.subheader("Market-relevant Probabilities")
        from soccer_prediction.markets.base import MarketDefinition
        from soccer_prediction.markets.evaluator import MarketEvaluator

        evaluator = MarketEvaluator(model=model)
        stats_markets = [
            MarketDefinition("over25", f"Over 2.5 Goals", "over_under_goals",
                             {"home_team": home, "away_team": away, "threshold": 2.5, "direction": "over", "is_neutral": is_neutral}),
            MarketDefinition("under25", f"Under 2.5 Goals", "over_under_goals",
                             {"home_team": home, "away_team": away, "threshold": 2.5, "direction": "under", "is_neutral": is_neutral}),
            MarketDefinition("btts", "Both Teams to Score", "both_teams_score",
                             {"home_team": home, "away_team": away, "is_neutral": is_neutral}),
            MarketDefinition("cs_home", f"{home} Clean Sheet", "clean_sheet",
                             {"team": home, "home_team": home, "away_team": away, "is_neutral": is_neutral}),
            MarketDefinition("cs_away", f"{away} Clean Sheet", "clean_sheet",
                             {"team": away, "home_team": home, "away_team": away, "is_neutral": is_neutral}),
        ]
        rows = [{"Market": mr.market.name, "Probability": f"{mr.model_probability:.1%}"}
                for mr in evaluator.evaluate_all(stats_markets)]
        st.table(pd.DataFrame(rows))


# ─── Tournament Simulation ────────────────────────────────────────────────────

def show_tournament_simulation():
    st.title("Tournament Simulation")
    st.caption("Monte Carlo simulation of the FIFA World Cup 2026")

    reg: ModelRegistry = st.session_state["model_registry"]
    model = reg.get(selected_model)
    if model is None:
        st.warning("No model trained. Click 'Train model' in the sidebar first.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        n_sims = st.select_slider(
            "Number of simulations",
            options=[1_000, 2_000, 5_000, 10_000, 25_000, 50_000],
            value=10_000,
        )
    with col2:
        seed = st.number_input("Random seed", value=42, min_value=0)

    if st.button("Run Simulation", type="primary"):
        groups = load_group_assignments()
        config = TournamentConfig.world_cup_2026(groups)

        progress = st.progress(0, text="Running simulations...")

        def on_progress(i, n):
            pct = i / n
            progress.progress(pct, text=f"Running simulations... {i:,}/{n:,}")

        sim = MonteCarloSimulator(n_simulations=n_sims, seed=int(seed))
        results = sim.run(config, model, progress_callback=on_progress)
        progress.empty()

        st.session_state["sim_results"] = results
        st.success(f"Completed {n_sims:,} simulations.")

    sim_results: SimulationResults = st.session_state.get("sim_results")
    if sim_results is None:
        st.info("Run a simulation to see results.")
        return

    probs_df = sim_results.team_probabilities()

    tab1, tab2, tab3 = st.tabs(["Tournament Probabilities", "Group Probabilities", "Export"])

    with tab1:
        st.subheader("Team Tournament Probabilities")

        stage_cols = [c for c in probs_df.columns if c not in ("team",) and probs_df[c].max() > 0]
        display_cols = ["team"] + [c for c in [
            "wins_tournament", "reaches_final", "reaches_semifinal",
            "reaches_quarterfinal", "reaches_round_of_16", "reaches_round_of_32",
            "advances_from_group", "group_winner",
        ] if c in stage_cols]

        show_df = probs_df[display_cols].copy()
        for col in display_cols[1:]:
            show_df[col] = show_df[col].map(lambda x: f"{x:.1%}")

        st.dataframe(show_df, use_container_width=True, height=500)

        st.subheader("Tournament Winner Probability (Top 16)")
        top16 = probs_df.head(16).copy()
        fig = px.bar(
            top16,
            x="team", y="wins_tournament",
            color="wins_tournament",
            color_continuous_scale="Blues",
            labels={"wins_tournament": "Win Probability", "team": "Team"},
            text=top16["wins_tournament"].map(lambda x: f"{x:.1%}"),
        )
        fig.update_layout(
            xaxis_tickangle=-35, showlegend=False,
            yaxis_tickformat=".0%", height=380, margin=dict(t=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Stage-Reach Probabilities (Top 12)")
        heat_cols = [c for c in [
            "advances_from_group", "reaches_round_of_16",
            "reaches_quarterfinal", "reaches_semifinal", "reaches_final", "wins_tournament",
        ] if c in probs_df.columns]
        heat_df = probs_df.head(12).set_index("team")[heat_cols]
        heat_labels = {
            "advances_from_group": "Adv. Group",
            "reaches_round_of_16": "RO16",
            "reaches_quarterfinal": "QF",
            "reaches_semifinal": "SF",
            "reaches_final": "Final",
            "wins_tournament": "Winner",
        }
        heat_df.columns = [heat_labels.get(c, c) for c in heat_df.columns]
        fig_h = px.imshow(
            heat_df.values * 100,
            x=heat_df.columns.tolist(),
            y=heat_df.index.tolist(),
            color_continuous_scale="Blues",
            text_auto=".0f",
            labels={"color": "Prob %"},
        )
        fig_h.update_layout(height=450, margin=dict(t=10))
        st.plotly_chart(fig_h, use_container_width=True)

    with tab2:
        st.subheader("Group Winner Probabilities")
        gw_df = sim_results.group_winner_probabilities()
        groups_list = sorted(gw_df["group"].unique())
        n_cols = 3
        rows = [groups_list[i:i+n_cols] for i in range(0, len(groups_list), n_cols)]
        for row_groups in rows:
            cols = st.columns(len(row_groups))
            for col, grp in zip(cols, row_groups):
                sub = gw_df[gw_df["group"] == grp].sort_values("prob_group_winner", ascending=False)
                with col:
                    st.markdown(f"**Group {grp}**")
                    for _, r in sub.iterrows():
                        st.write(f"  {r['team']}: {r['prob_group_winner']:.1%}")

    with tab3:
        st.subheader("Export Results")
        csv = probs_df.to_csv(index=False)
        st.download_button(
            "Download team probabilities (CSV)",
            data=csv,
            file_name="wc2026_team_probabilities.csv",
            mime="text/csv",
        )
        mkt_df = sim_results.match_probabilities()
        if not mkt_df.empty:
            st.download_button(
                "Download match probabilities (CSV)",
                data=mkt_df.to_csv(index=False),
                file_name="wc2026_match_probabilities.csv",
                mime="text/csv",
            )


# ─── Market Analysis ──────────────────────────────────────────────────────────

def show_market_analysis():
    st.title("Market Analysis")
    st.caption("Compare model probabilities to Kalshi/market prices")

    mkt_reg: MarketRegistry = st.session_state["market_registry"]
    if len(mkt_reg) == 0:
        n = mkt_reg.load_directory(MARKETS_DIR)
        st.info(f"Loaded {n} markets from configs/markets/")

    reg: ModelRegistry = st.session_state["model_registry"]
    model = reg.get(selected_model)
    sim_results: SimulationResults = st.session_state.get("sim_results")

    if model is None and sim_results is None:
        st.warning("Train a model and/or run a tournament simulation to evaluate markets.")
        return

    evaluator = MarketEvaluator(sim_results=sim_results, model=model)
    all_markets = mkt_reg.all()

    if not all_markets:
        st.warning("No markets found. Check configs/markets/ directory.")
        return

    market_results_list = evaluator.evaluate_all(all_markets)

    rows = [mr.to_dict() for mr in market_results_list]
    df = pd.DataFrame(rows)
    df["model_probability"] = df["model_probability"].where(df["model_probability"].notna(), other=None)

    categories = ["All"] + sorted(df["category"].unique().tolist())
    cat_filter = st.selectbox("Filter by category", categories)
    if cat_filter != "All":
        df = df[df["category"] == cat_filter]

    st.subheader("Market Probabilities")

    # Allow user to input market prices inline
    st.caption("Enter Kalshi prices (0–100) in the 'Market Price (%)' column to compute edge.")

    display_df = df[["name", "category", "model_probability", "market_price", "edge"]].copy()
    display_df["model_probability"] = display_df["model_probability"].map(
        lambda x: f"{x:.1%}" if pd.notna(x) else "—"
    )
    display_df["market_price"] = display_df["market_price"].map(
        lambda x: f"{x:.1%}" if pd.notna(x) else "—"
    )
    display_df["edge"] = display_df["edge"].map(
        lambda x: f"{x:+.1%}" if pd.notna(x) else "—"
    )
    display_df.columns = ["Market", "Category", "Model Prob", "Market Price", "Edge"]
    st.dataframe(display_df, use_container_width=True, height=500)

    # Edge chart for markets with known prices
    edge_df = df[(df["market_price"].notna()) & (df["model_probability"].notna())].copy()
    if not edge_df.empty:
        edge_df["edge_num"] = edge_df["model_probability"].astype(float) - edge_df["market_price"].astype(float)
        edge_df = edge_df.sort_values("edge_num", ascending=False)
        fig = px.bar(
            edge_df,
            x="name", y="edge_num",
            color="edge_num",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            title="Model Edge vs Market (positive = model favours this market)",
            labels={"edge_num": "Edge", "name": "Market"},
        )
        fig.update_layout(xaxis_tickangle=-35, yaxis_tickformat=".0%", height=380, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Update Market Prices")
    st.caption("Paste in the current Kalshi implied probability for any market.")
    with st.form("price_update"):
        market_ids = [m.id for m in all_markets]
        sel_market = st.selectbox("Market", options=market_ids,
                                  format_func=lambda mid: mkt_reg.get(mid).name)
        new_price = st.number_input("Kalshi price (0.0 – 1.0)", min_value=0.0, max_value=1.0, step=0.01, value=0.5)
        if st.form_submit_button("Update"):
            mkt_reg.update_price(sel_market, new_price)
            st.success(f"Updated price for '{sel_market}'")
            st.rerun()

    st.download_button(
        "Export market analysis (CSV)",
        data=df.to_csv(index=False),
        file_name="market_analysis.csv",
        mime="text/csv",
    )


# ─── Feature Inspector ────────────────────────────────────────────────────────

def show_feature_inspector():
    st.title("Feature Inspector")

    reg: ModelRegistry = st.session_state["model_registry"]
    model = reg.get(selected_model)

    if model is None:
        st.warning("Train a model first.")
        return

    teams_df = load_teams_df()
    if teams_df.empty:
        st.warning("No teams loaded.")
        return

    # Elo ratings table
    if hasattr(model, "get_ratings"):
        ratings = model.get_ratings()
        elo_df = (
            pd.DataFrame(ratings.items(), columns=["team", "elo_rating"])
            .sort_values("elo_rating", ascending=False)
            .reset_index(drop=True)
        )
        elo_df.insert(0, "rank", range(1, len(elo_df) + 1))

        st.subheader("Current Elo Ratings")
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.bar(
                elo_df.head(20), x="team", y="elo_rating",
                labels={"elo_rating": "Elo Rating", "team": "Team"},
                color="elo_rating", color_continuous_scale="Blues",
                height=350,
            )
            fig.update_layout(xaxis_tickangle=-40, margin=dict(t=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.dataframe(elo_df, use_container_width=True, height=400)

    # Poisson attack/defense table
    if hasattr(model, "get_attack_defense_table"):
        st.subheader("Poisson Attack / Defense Parameters")
        ad_df = model.get_attack_defense_table()
        st.dataframe(ad_df, use_container_width=True)

        fig = px.scatter(
            ad_df, x="defense", y="attack", text="team",
            labels={"attack": "Attack Strength (relative)", "defense": "Defense Weakness (relative)"},
            height=450,
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color="grey")
        fig.add_vline(x=1.0, line_dash="dash", line_color="grey")
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

    # Single match features
    st.subheader("Match Feature Snapshot")
    team_names = sorted(teams_df["name"].tolist())
    c1, c2 = st.columns(2)
    with c1:
        home_f = st.selectbox("Home team", team_names, key="feat_home")
    with c2:
        away_f = st.selectbox("Away team", team_names,
                               index=1 if len(team_names) > 1 else 0, key="feat_away")
    if st.button("Show features"):
        if hasattr(model, "elo"):
            feats = model.elo.get_match_features(home_f, away_f, is_neutral=True)
            st.json(feats)


# ─── Data Manager ─────────────────────────────────────────────────────────────

def show_data_manager():
    st.title("Data Manager")

    tab1, tab2, tab3 = st.tabs(["Teams", "Historical Matches", "Fixtures"])

    with tab1:
        teams_df = load_teams_df()
        st.metric("Teams in database", len(teams_df))
        if not teams_df.empty:
            conf_filter = st.selectbox(
                "Filter by confederation",
                ["All"] + sorted(teams_df["confederation"].dropna().unique()),
            )
            df = teams_df if conf_filter == "All" else teams_df[teams_df["confederation"] == conf_filter]
            st.dataframe(df, use_container_width=True)

        st.subheader("Upload teams CSV")
        uploaded = st.file_uploader("Upload teams CSV", type=["csv"], key="teams_upload")
        if uploaded:
            import tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
                f.write(uploaded.read())
                tmp = f.name
            loader = DataLoader()
            n = loader.load_teams(Path(tmp))
            os.unlink(tmp)
            invalidate_data_caches()
            st.success(f"Loaded {n} new teams.")

    with tab2:
        matches_df = load_matches_df()
        st.metric("Historical matches", len(matches_df))
        if not matches_df.empty:
            st.dataframe(matches_df.tail(50), use_container_width=True)

        st.subheader("Upload historical matches CSV")
        up_hist = st.file_uploader("Upload historical matches CSV", type=["csv"], key="hist_upload")
        if up_hist:
            import tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
                f.write(up_hist.read())
                tmp = f.name
            loader = DataLoader()
            n = loader.load_historical_matches(Path(tmp))
            os.unlink(tmp)
            invalidate_data_caches()
            st.success(f"Loaded {n} matches.")

    with tab3:
        fixtures_df = load_fixtures_df()
        st.metric("WC 2026 fixtures", len(fixtures_df))
        if not fixtures_df.empty:
            groups = ["All"] + sorted(fixtures_df["group_name"].dropna().unique())
            grp_filter = st.selectbox("Filter by group", groups)
            df = fixtures_df if grp_filter == "All" else fixtures_df[fixtures_df["group_name"] == grp_filter]
            st.dataframe(df, use_container_width=True)

        st.subheader("Upload fixtures CSV")
        up_fix = st.file_uploader("Upload fixtures CSV", type=["csv"], key="fix_upload")
        if up_fix:
            import tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
                f.write(up_fix.read())
                tmp = f.name
            loader = DataLoader()
            n = loader.load_fixtures(Path(tmp), "FIFA World Cup", 2026)
            os.unlink(tmp)
            invalidate_data_caches()
            st.success(f"Loaded {n} fixtures.")


# ─── Router ───────────────────────────────────────────────────────────────────

if page == "Overview":
    show_overview()
elif page == "Match Predictor":
    show_match_predictor()
elif page == "Tournament Simulation":
    show_tournament_simulation()
elif page == "Market Analysis":
    show_market_analysis()
elif page == "Feature Inspector":
    show_feature_inspector()
elif page == "Data Manager":
    show_data_manager()
