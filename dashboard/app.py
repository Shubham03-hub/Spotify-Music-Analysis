"""
Spotify Track Performance Intelligence — Streamlit Dashboard (PHASE 10).

This is a pure consumption layer: it imports src.prediction.predict for
scoring and reads pre-computed CSVs/figures from data/processed and
reports/. It contains no modeling logic of its own, by design (see
architecture notes in docs/) — retraining always happens through
src.training, never inside the dashboard process.

Run with: streamlit run dashboard/app.py
"""

import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.prediction.predict import predict_tier_probabilities, score_tracks
from src.utils.helper import load_config, resolve_path
from src.utils.model_loader import load_model_metadata, models_are_available

st.set_page_config(
    page_title="Spotify Track Performance Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",

)

# ---------------------------------------------------------------------------
# Crextio-inspired light styling (cream/gold, soft cards, black pill buttons)
# ---------------------------------------------------------------------------
st.markdown("""
<style>

/* ---------- App background ---------- */
.stApp {
    background: linear-gradient(160deg, #F8F3E7 0%, #FBEECB 55%, #F5E9CE 100%);
    color: #1A1A1A;
}

/* ---------- Titles ---------- */
h1, h2, h3 {
    color: #1A1A1A !important;
    font-weight: 700 !important;
    border-left: 4px solid #E8B93A;
    padding-left: 12px;
    margin-bottom: 0.6em !important;
}

h1 {
    font-size: 2.2rem !important;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #ECE3D0;
}

section[data-testid="stSidebar"] * {
    color: #4A4A4A !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #1A1A1A !important;
    border-left: 3px solid #E8B93A;
}

/* ---------- Metric / card containers ---------- */
div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border-radius: 20px;
    padding: 18px 22px;
    border: 1px solid #F0E7D2;
    box-shadow: 0 6px 18px rgba(26, 26, 26, 0.06);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}

div[data-testid="stMetric"]:hover {
    box-shadow: 0 10px 26px rgba(26, 26, 26, 0.10);
    transform: translateY(-2px);
}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] p {
    color: #6E6E6E !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}

div[data-testid="stMetricValue"] {
    color: #1A1A1A !important;
    font-weight: 800 !important;
}

.spotify-card {
    background-color: #FFFFFF;
    border-radius: 20px;
    padding: 20px;
    border: 1px solid #F0E7D2;
    box-shadow: 0 6px 18px rgba(26, 26, 26, 0.06);
    margin-bottom: 16px;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF;
    border-radius: 20px !important;
    border: 1px solid #F0E7D2 !important;
    box-shadow: 0 6px 18px rgba(26, 26, 26, 0.05);
}

/* ---------- Buttons — solid black pill, like the Crextio nav ---------- */
.stButton > button {
    background-color: #1A1A1A;
    color: #FFFFFF;
    border: none;
    border-radius: 500px; /* pill shape */
    padding: 10px 32px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-size: 0.8rem;
    transition: all 0.2s ease-in-out;
}

.stButton > button:hover {
    background-color: #333333;
    transform: scale(1.03);
    color: #F2C94C;
}

.stButton > button:active {
    background-color: #000000;
    transform: scale(0.98);
}

/* ---------- Tabs — pill-style like the top nav ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: none;
    background-color: #FFFFFF;
    padding: 6px;
    border-radius: 500px;
    border: 1px solid #F0E7D2;
    width: fit-content;
}

.stTabs [data-baseweb="tab"] {
    color: #6E6E6E;
    font-weight: 600;
    border-radius: 500px;
    padding: 6px 18px;
}

.stTabs [aria-selected="true"] {
    color: #FFFFFF !important;
    background-color: #1A1A1A !important;
    border-bottom: none !important;
}

/* ---------- Dataframes / tables ---------- */
div[data-testid="stDataFrame"] {
    background-color: #FFFFFF;
    border-radius: 14px;
    border: 1px solid #F0E7D2;
}

/* ---------- Selectbox / inputs ---------- */
.stSelectbox > div > div,
.stTextInput > div > div,
.stMultiSelect > div > div {
    background-color: #FFFFFF;
    border: 1px solid #E7DCC0;
    color: #1A1A1A;
    border-radius: 12px;
}

/* ---------- Contrast fixes: captions, widget labels, markdown text ---------- */
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #7A7A7A !important;
    opacity: 1 !important;
}

[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
    color: #1A1A1A !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}

.stMarkdown p, .stMarkdown li, .stMarkdown span {
    color: #2E2E2E;
}

.stTabs [data-baseweb="tab"] p {
    color: #6E6E6E !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}

.stTabs [aria-selected="true"] p {
    color: #FFFFFF !important;
}

/* ---------- Number inputs ---------- */
.stNumberInput input,
.stNumberInput > div > div {
    background-color: #FFFFFF !important;
    color: #1A1A1A !important;
    border: 1px solid #E7DCC0 !important;
    border-radius: 12px !important;
}

/* ---------- Sliders — warm gold fill ---------- */
div[data-testid="stSlider"] [role="slider"] {
    background-color: #E8B93A !important;
    border-color: #E8B93A !important;
    box-shadow: 0 0 0 4px rgba(232, 185, 58, 0.25) !important;
}
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
    background-color: #E8B93A !important;
}
div[data-testid="stSlider"] div[data-baseweb="slider"] > div:first-child {
    background-color: #EAE0C7 !important;
}

/* ---------- Checkbox accent ---------- */
.stCheckbox [data-baseweb="checkbox"] svg {
    fill: #E8B93A !important;
}

/* ---------- Download button (separate testid from regular button) ---------- */
.stDownloadButton > button {
    background-color: #1A1A1A;
    color: #FFFFFF;
    border: none;
    border-radius: 500px;
    padding: 10px 32px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-size: 0.8rem;
}
.stDownloadButton > button:hover {
    background-color: #333333;
    color: #F2C94C;
}

/* ---------- Alert boxes (success/warning/error) ---------- */
div[data-testid="stAlert"] {
    background-color: #FFFFFF;
    border-radius: 14px;
    border: 1px solid #F0E7D2;
}

/* ---------- Scrollbar ---------- */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: #F5E9CE;
}
::-webkit-scrollbar-thumb {
    background: #D8C79A;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #E8B93A;
}

</style>
""", unsafe_allow_html=True)

CONFIG = load_config()
PROCESSED_DIR = resolve_path(CONFIG["paths"]["processed_dir"])
REPORTS_DIR = resolve_path(CONFIG["paths"]["reports_dir"])
FIGURES_DIR = resolve_path(CONFIG["paths"]["figures_dir"])


@st.cache_data
def load_catalog() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "model_ready_dataset.csv")


@st.cache_data
def load_genre_year() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "genre_year_aggregates.csv")


@st.cache_data
def load_leaderboards():
    reg = pd.read_csv(REPORTS_DIR / "regression_leaderboard.csv")
    clf = pd.read_csv(REPORTS_DIR / "classification_leaderboard.csv")
    return reg, clf


@st.cache_data
def load_driver_importance():
    reg_imp = pd.read_csv(REPORTS_DIR / "driver_importance_regression.csv")
    clf_imp = pd.read_csv(REPORTS_DIR / "driver_importance_classification.csv")
    return reg_imp, clf_imp


def clean_feature_label(raw_name: str) -> str:
    """Turn 'num__artist_artist_star_power' into 'Artist Star Power' for display."""
    name = raw_name.split("__", 1)[-1]
    return name.replace("_", " ").replace("cat ", "").title()


TIER_COLORS = {
    "Flop": "#B0B0B0", "Niche": "#5DA9E9", "Mid": "#F2C94C",
    "Hit": "#F2994A", "Viral Hit": "#EB5757",
}

GOLD_SCALE = ["#FFFFFF", "#FCEFD0", "#F2C94C", "#E8B93A", "#8A6A15"]

PLOTLY_LIGHT_LAYOUT = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    font=dict(color="#1A1A1A", family="Helvetica, Arial, sans-serif"),
    title_font=dict(color="#1A1A1A", size=16),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#1A1A1A")),
    xaxis=dict(gridcolor="#F0E7D2", zerolinecolor="#F0E7D2", color="#6E6E6E"),
    yaxis=dict(gridcolor="#F0E7D2", zerolinecolor="#F0E7D2", color="#6E6E6E"),
    margin=dict(t=50, l=10, r=10, b=10),
)


def themed(fig):
    """Apply consistent cream/gold styling to any Plotly figure."""
    fig.update_layout(**PLOTLY_LIGHT_LAYOUT)
    return fig


def main():
    st.markdown("""
    <div style="display:flex;align-items:center;gap:16px;padding:4px 0 6px 0;">
        <div style="width:52px;height:52px;border-radius:50%;background:#1A1A1A;
                    display:flex;align-items:center;justify-content:center;flex-shrink:0;
                    box-shadow:0 4px 16px rgba(26,26,26,0.20);">
            <span style="font-size:26px;">🎵</span>
        </div>
        <div>
            <div style="font-size:2rem;font-weight:800;color:#1A1A1A;line-height:1.15;">
                Spotify Track Performance Intelligence
            </div>
            <div style="font-size:0.95rem;color:#7A7A7A;margin-top:2px;">
                Track popularity scoring, success-tier classification, and genre trend
                forecasting for A&amp;R, marketing, and catalog strategy.
            </div>
        </div>
    </div>
    <hr style="border:none;border-top:1px solid #EEE3C8;margin:16px 0 6px 0;">
    """, unsafe_allow_html=True)

    if not models_are_available(CONFIG):
        st.error(
            "No trained models found in `models/`. Run "
            "`python -m src.training.mlflow_tracking` before launching this dashboard."
        )
        st.stop()

    catalog = load_catalog()
    genre_year = load_genre_year()
    reg_leaderboard, clf_leaderboard = load_leaderboards()
    reg_importance, clf_importance = load_driver_importance()
    metadata = load_model_metadata(CONFIG)

    # -------------------------------------------------------------------
    # Sidebar filters — genre selector, country selector, indicator selector
    # -------------------------------------------------------------------
    st.sidebar.header("Filters")
    all_genres = sorted(catalog["genre"].unique())
    selected_genres = st.sidebar.multiselect("Genre", all_genres, default=all_genres)

    all_countries = sorted(catalog["artist_country"].unique())
    selected_countries = st.sidebar.multiselect("Artist Country", all_countries, default=all_countries)

    indicator_options = [
        "danceability", "energy", "valence", "acousticness", "tempo",
        "loudness", "speechiness", "instrumentalness", "liveness",
    ]
    selected_indicator = st.sidebar.selectbox("Audio Indicator to Explore", indicator_options, index=0)

    filtered = catalog[
        catalog["genre"].isin(selected_genres) & catalog["artist_country"].isin(selected_countries)
    ]
    if filtered.empty:
        st.warning("No tracks match the current filters.")
        st.stop()

    # -------------------------------------------------------------------
    # Welcome banner + quick-stat pills
    # -------------------------------------------------------------------
    genre_coverage = len(selected_genres) / len(all_genres) if all_genres else 0
    artist_coverage = filtered["artist_id"].nunique() / catalog["artist_id"].nunique()
    hit_rate_pct = filtered["popularity_tier"].isin(["Hit", "Viral Hit"]).mean()

    st.markdown(f"""
    <div style="margin:4px 0 20px 0;">
        <div style="font-size:1.5rem;font-weight:700;color:#1A1A1A;margin-bottom:12px;">
            Welcome back — here's how the catalog looks
        </div>
        <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;">
            <span style="background:#1A1A1A;color:#FFFFFF;padding:8px 18px;border-radius:500px;
                         font-size:0.85rem;font-weight:600;">
                Genre Coverage {genre_coverage:.0%}
            </span>
            <span style="background:#F2C94C;color:#1A1A1A;padding:8px 18px;border-radius:500px;
                         font-size:0.85rem;font-weight:700;">
                Hit + Viral Rate {hit_rate_pct:.0%}
            </span>
            <span style="background:#FFFFFF;color:#1A1A1A;border:1px solid #E7DCC0;padding:8px 18px;
                         border-radius:500px;font-size:0.85rem;font-weight:600;">
                Artist Coverage {artist_coverage:.0%}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------
    # Executive Summary + KPI cards
    # -------------------------------------------------------------------
    st.subheader("Executive Summary")
    st.markdown(
        f"Catalog of **{len(catalog):,} tracks** across **{catalog['genre'].nunique()} genres** and "
        f"**{catalog['artist_id'].nunique()} artists** (2020-2026). Regression model "
        f"(`{metadata['regression_model']}`) explains "
        f"**{reg_leaderboard.loc[reg_leaderboard['model'] == metadata['regression_model'], 'r2_mean'].values[0]:.0%}** "
        f"of popularity variance; classification model (`{metadata['classification_model']}`) reaches "
        f"**{clf_leaderboard.loc[clf_leaderboard['model'] == metadata['classification_model'], 'f1_macro_mean'].values[0]:.0%} macro-F1** "
        f"on success tier, despite severe class imbalance (Viral Hit is <0.3% of the catalog)."
    )

    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Tracks (filtered)", f"{len(filtered):,}")
    kpi_cols[1].metric("Avg Popularity", f"{filtered['popularity'].mean():.1f}")
    kpi_cols[2].metric("Hit + Viral Hit Rate", f"{filtered['popularity_tier'].isin(['Hit', 'Viral Hit']).mean():.1%}")
    top_genre = filtered.groupby("genre")["popularity"].mean().idxmax()
    kpi_cols[3].metric("Top Genre (avg pop.)", top_genre.replace("_", " ").title())
    kpi_cols[4].metric("Artists Represented", f"{filtered['artist_id'].nunique()}")

    st.markdown("##### Catalog Composition")
    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        tier_counts = (
            filtered["popularity_tier"].value_counts()
            .reindex(TIER_COLORS.keys()).fillna(0).reset_index()
        )
        tier_counts.columns = ["tier", "count"]
        fig_donut = px.pie(
            tier_counts, names="tier", values="count", hole=0.55,
            title="Success Tier Distribution",
            color="tier", color_discrete_map=TIER_COLORS,
        )
        fig_donut.update_traces(textinfo="percent+label", textfont_color="#FFFFFF")
        st.plotly_chart(themed(fig_donut), use_container_width=True)
    with ec2:
        fig_hist = px.histogram(
            filtered, x="popularity", nbins=30, color="popularity_tier",
            color_discrete_map=TIER_COLORS, title="Popularity Score Distribution",
        )
        fig_hist.update_layout(bargap=0.05)
        st.plotly_chart(themed(fig_hist), use_container_width=True)
    with ec3:
        gauge_max = max(20, hit_rate_pct * 100 * 2)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=hit_rate_pct * 100,
            number={"suffix": "%", "font": {"color": "#1A1A1A", "size": 34}},
            title={"text": "Hit + Viral Hit Rate", "font": {"color": "#1A1A1A", "size": 14}},
            gauge={
                "axis": {"range": [0, gauge_max], "tickcolor": "#9A9A9A"},
                "bar": {"color": "#E8B93A", "thickness": 0.3},
                "bgcolor": "#FFFFFF",
                "borderwidth": 0,
                "steps": [{"range": [0, gauge_max], "color": "#F5E9CE"}],
            },
        ))
        st.plotly_chart(themed(fig_gauge), use_container_width=True)

    st.divider()

    tab_scorer, tab_trends, tab_drivers, tab_catalog = st.tabs(
        ["🎯 Track Scorer", "📈 Genre Trends & Forecast", "🔑 Driver Importance", "📋 Catalog & Export"]
    )

    # -------------------------------------------------------------------
    # TAB 1: Track scorer — score a hypothetical/new track interactively
    # -------------------------------------------------------------------
    with tab_scorer:
        st.markdown("Score a track (new or hypothetical) using its audio features and artist context.")
        col1, col2, col3 = st.columns(3)
        with col1:
            genre = st.selectbox("Genre", all_genres)
            danceability = st.slider("Danceability", 0.0, 1.0, 0.65)
            energy = st.slider("Energy", 0.0, 1.0, 0.65)
            valence = st.slider("Valence (mood positivity)", 0.0, 1.0, 0.5)
        with col2:
            acousticness = st.slider("Acousticness", 0.0, 1.0, 0.2)
            instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.05)
            speechiness = st.slider("Speechiness", 0.0, 1.0, 0.08)
            liveness = st.slider("Liveness", 0.0, 1.0, 0.15)
        with col3:
            tempo = st.slider("Tempo (BPM)", 60, 200, 110)
            loudness = st.slider("Loudness (dB)", -30.0, 0.0, -7.0)
            artist_star_power = st.slider("Artist Star Power (0-100)", 0.0, 100.0, 40.0)
            artist_followers_millions = st.number_input("Artist Followers (millions)", 0.0, 200.0, 2.0)

        col4, col5, col6 = st.columns(3)
        with col4:
            duration_minutes_input = st.number_input("Duration (minutes)", 1.0, 12.0, 3.3)
        with col5:
            release_year = st.number_input("Release Year", 2020, 2027, 2026)
        with col6:
            is_explicit = st.checkbox("Explicit", value=False)

        if st.button("Score This Track", type="primary"):
            genre_row = catalog[catalog["genre"] == genre].iloc[0]
            input_row = genre_row.copy()
            input_row["genre"] = genre
            input_row["danceability"] = danceability
            input_row["energy"] = energy
            input_row["valence"] = valence
            input_row["acousticness"] = acousticness
            input_row["instrumentalness"] = instrumentalness
            input_row["speechiness"] = speechiness
            input_row["liveness"] = liveness
            input_row["tempo"] = tempo
            input_row["loudness"] = loudness
            input_row["artist_artist_star_power"] = artist_star_power
            input_row["artist_followers_millions"] = artist_followers_millions
            input_row["duration_minutes"] = duration_minutes_input
            input_row["duration_ms"] = duration_minutes_input * 60000
            input_row["release_year"] = release_year
            input_row["is_explicit_int"] = int(is_explicit)
            input_row["explicit"] = is_explicit
            input_row["energy_danceability"] = energy * danceability
            input_row["valence_energy"] = valence * energy
            input_row["acoustic_electronic_balance"] = acousticness - energy
            # No prior-release history for a hypothetical new track:
            # neutral/debut defaults, consistent with training-time fillna logic.
            input_row["artist_prior_track_popularity"] = catalog["popularity"].median()
            input_row["artist_prior_avg_popularity"] = catalog["popularity"].median()
            input_row["artist_prior_track_count"] = 0
            input_row["artist_prior_popularity_std"] = 0.0
            input_row["is_artist_debut_track"] = 1

            input_df = pd.DataFrame([input_row])
            scored = score_tracks(input_df, CONFIG)

            r1, r2 = st.columns(2)
            with r1:
                st.metric("Predicted Popularity Score", f"{scored['predicted_popularity'].iloc[0]:.1f} / 100")
            with r2:
                predicted_tier = scored["predicted_tier"].iloc[0]
                st.metric("Predicted Success Tier", predicted_tier)

            prob_cols = [c for c in scored.columns if c.startswith("prob_")]
            probs = scored[prob_cols].iloc[0]
            probs.index = [c.replace("prob_", "").replace("_", " ") for c in probs.index]
            fig = px.bar(
                x=probs.values, y=probs.index, orientation="h",
                labels={"x": "Probability", "y": "Tier"},
                title="Tier Probability Breakdown",
                color=probs.index,
                color_discrete_map={k: v for k, v in TIER_COLORS.items()},
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(themed(fig), use_container_width=True)

            audio_features = [
                "danceability", "energy", "valence", "acousticness",
                "instrumentalness", "speechiness", "liveness",
            ]
            track_values = [
                danceability, energy, valence, acousticness,
                instrumentalness, speechiness, liveness,
            ]
            genre_avg = catalog[catalog["genre"] == genre][audio_features].mean().tolist()
            theta_labels = [f.title() for f in audio_features]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=track_values + track_values[:1],
                theta=theta_labels + theta_labels[:1],
                fill="toself", name="This Track",
                line_color="#E8B93A", fillcolor="rgba(232,185,58,0.35)",
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=genre_avg + genre_avg[:1],
                theta=theta_labels + theta_labels[:1],
                fill="toself", name=f"{genre.replace('_', ' ').title()} Avg",
                line_color="#9A9A9A", fillcolor="rgba(154,154,154,0.15)",
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="#FFFFFF",
                    radialaxis=dict(visible=True, range=[0, 1], color="#9A9A9A", gridcolor="#F0E7D2"),
                    angularaxis=dict(color="#1A1A1A"),
                ),
                title="Audio Feature Profile vs. Genre Average",
                showlegend=True,
            )
            st.plotly_chart(themed(fig_radar), use_container_width=True)

    # -------------------------------------------------------------------
    # TAB 2: Genre trends + simple forward forecast
    # -------------------------------------------------------------------
    with tab_trends:
        st.markdown(f"Genre-year average popularity trend, with selected indicator: **{selected_indicator}**")
        gy_filtered = genre_year[genre_year["genre"].isin(selected_genres)]

        fig = px.line(
            gy_filtered, x="release_year", y="avg_popularity", color="genre", markers=True,
            title="Average Popularity by Genre (2020-2026)",
        )
        st.plotly_chart(themed(fig), use_container_width=True)

        st.markdown("#### One-Year-Ahead Forecast (trend-slope extrapolation)")
        st.caption(
            "Forecast = last observed genre-year average popularity + that genre's own linear "
            "trend slope. A simple, transparent baseline — not a substitute for the ML regression "
            "model, which scores individual tracks rather than genre aggregates."
        )
        latest_year = gy_filtered["release_year"].max()
        latest = gy_filtered[gy_filtered["release_year"] == latest_year].copy()
        latest["forecast_next_year"] = latest["avg_popularity"] + latest["popularity_trend_slope"]
        latest["direction"] = np.where(latest["popularity_trend_slope"] > 0, "rising", "declining")

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=latest["genre"], y=latest["avg_popularity"], name=f"{latest_year} actual",
            marker_color="#E8B93A",
        ))
        fig2.add_trace(go.Bar(
            x=latest["genre"], y=latest["forecast_next_year"], name=f"{latest_year + 1} forecast",
            marker_color="#4CD97B",
        ))
        fig2.update_layout(barmode="group", title="Latest Year vs Forecast Next Year, by Genre")
        st.plotly_chart(themed(fig2), use_container_width=True)

        st.dataframe(
            latest[["genre", "avg_popularity", "popularity_trend_slope", "forecast_next_year", "direction"]]
            .sort_values("popularity_trend_slope", ascending=False)
            .rename(columns={
                "avg_popularity": f"{latest_year} avg popularity",
                "popularity_trend_slope": "trend slope",
                "forecast_next_year": f"{latest_year + 1} forecast",
            }),
            use_container_width=True, hide_index=True,
        )

        fig3 = px.box(
            filtered, x="genre", y=selected_indicator,
            color_discrete_sequence=["#E8B93A"],
            title=f"{selected_indicator.title()} Distribution by Genre",
        )
        st.plotly_chart(themed(fig3), use_container_width=True)

        st.markdown("#### Energy vs. Valence Landscape")
        scatter_df = filtered if len(filtered) <= 3000 else filtered.sample(3000, random_state=42)
        fig4 = px.scatter(
            scatter_df, x="energy", y="valence", color="popularity_tier",
            size="popularity", size_max=14, opacity=0.75,
            hover_data=["track_name", "genre"],
            color_discrete_map=TIER_COLORS,
            title="Energy vs. Valence, sized by Popularity",
        )
        st.plotly_chart(themed(fig4), use_container_width=True)

    # -------------------------------------------------------------------
    # TAB 3: Driver importance
    # -------------------------------------------------------------------
    with tab_drivers:
        d1, d2 = st.columns(2)
        with d1:
            st.markdown(f"##### Popularity Score Drivers (`{metadata['regression_model']}`)")
            reg_importance_display = reg_importance.copy()
            reg_importance_display["feature"] = reg_importance_display["feature"].apply(clean_feature_label)
            fig = px.bar(
                reg_importance_display.head(12), x="importance", y="feature", orientation="h",
                color_discrete_sequence=["#E8B93A"],
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(themed(fig), use_container_width=True)
        with d2:
            st.markdown(f"##### Success Tier Drivers (`{metadata['classification_model']}`)")
            clf_importance_display = clf_importance.copy()
            clf_importance_display["feature"] = clf_importance_display["feature"].apply(clean_feature_label)
            fig = px.bar(
                clf_importance_display.head(12), x="importance", y="feature", orientation="h",
                color_discrete_sequence=["#E8B93A"],
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(themed(fig), use_container_width=True)

        st.markdown("##### Audio Feature Correlation with Popularity")
        audio_cols = [
            "danceability", "energy", "valence", "acousticness", "instrumentalness",
            "speechiness", "liveness", "tempo", "loudness", "popularity",
        ]
        corr = filtered[audio_cols].corr().round(2)
        fig_heat = px.imshow(
            corr, text_auto=True, color_continuous_scale=GOLD_SCALE,
            aspect="auto", title="Feature Correlation Matrix",
        )
        st.plotly_chart(themed(fig_heat), use_container_width=True)

        st.markdown("##### Model Leaderboards (5-fold cross-validated)")
        l1, l2 = st.columns(2)
        l1.dataframe(reg_leaderboard.round(3), use_container_width=True, hide_index=True)
        l2.dataframe(clf_leaderboard.round(3), use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------
    # TAB 4: Catalog table + download predictions
    # -------------------------------------------------------------------
    with tab_catalog:
        top5 = filtered.sort_values("popularity", ascending=False).head(5)
        rows_html = ""
        for _, row in top5.iterrows():
            track_name = html.escape(str(row["track_name"]))
            artist_name = html.escape(str(row["artist_artist_name"]))
            genre_label = html.escape(str(row["genre"]).replace("_", " ").title())
            rows_html += f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:10px 4px;border-bottom:1px solid #333333;">
                <div>
                    <div style="color:#FFFFFF;font-weight:600;font-size:0.9rem;">{track_name}</div>
                    <div style="color:#9A9A9A;font-size:0.78rem;">{artist_name} · {genre_label}</div>
                </div>
                <div style="background:#E8B93A;color:#1A1A1A;font-weight:700;font-size:0.8rem;
                            padding:4px 12px;border-radius:500px;flex-shrink:0;margin-left:12px;">
                    {row['popularity']:.0f}
                </div>
            </div>
            """
        st.markdown(f"""
        <div style="background:#1A1A1A;border-radius:20px;padding:20px 22px;margin:6px 0 22px 0;">
            <div style="color:#FFFFFF;font-weight:700;font-size:1.05rem;margin-bottom:6px;">
                🏆 Top Tracks Spotlight
            </div>
            {rows_html}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### Top Artists by Track Count (filtered)")
        top_artists = (
            filtered.groupby("artist_artist_name")["track_name"].count()
            .sort_values(ascending=False).head(10).reset_index()
        )
        top_artists.columns = ["artist", "track_count"]
        fig_artists = px.bar(
            top_artists, x="track_count", y="artist", orientation="h",
            color_discrete_sequence=["#E8B93A"], title="Top 10 Artists by Track Count",
        )
        fig_artists.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(themed(fig_artists), use_container_width=True)

        st.markdown(f"Showing {len(filtered):,} of {len(catalog):,} tracks matching current filters.")
        display_cols = [
            "track_name", "artist_artist_name", "genre", "release_year",
            "popularity", "popularity_tier", "artist_artist_star_power",
        ]
        st.dataframe(filtered[display_cols].sort_values("popularity", ascending=False), use_container_width=True, hide_index=True)

        st.markdown("##### Score & Download Full Catalog")
        st.caption("Runs both models against every track matching the current filters and produces a downloadable CSV.")
        if st.button("Generate Scored Predictions"):
            with st.spinner("Scoring filtered catalog..."):
                scored = score_tracks(filtered, CONFIG)
            st.success(f"Scored {len(scored)} tracks.")
            csv_bytes = scored.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download Predictions CSV", data=csv_bytes,
                file_name="spotify_scored_predictions.csv", mime="text/csv",
            )
            st.dataframe(
                scored[["track_name", "genre", "popularity", "predicted_popularity", "popularity_tier", "predicted_tier"]].head(50),
                use_container_width=True, hide_index=True,
            )


if __name__ == "__main__":
    main()
