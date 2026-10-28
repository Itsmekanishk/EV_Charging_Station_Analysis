"""ChargeMap Intelligence — EV charging business location planner.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from data_loader import AUSTIN_BBOX, CACHE_DIR, CITY_NAME, load_all_data
from map_viz import (
    create_cluster_map,
    create_opportunity_map,
    create_source_map,
)
from model import (
    CRITERIA_LABELS,
    DEFAULT_WEIGHTS,
    cluster_demand_zones,
    criterion_breakdown,
    normalise_weights,
    score_grid,
    shortlist_sites,
)

st.set_page_config(
    page_title="ChargeMap Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_SOURCES = {
    "OpenStreetMap · auto fallback": "auto",
    "Simulated · offline": "simulated",
}
TIER_ACCENTS = {"Prime": "#facc15", "Strong": "#22d3ee", "Moderate": "#818cf8", "Low": "#94a3b8"}
WEIGHT_LABELS = {
    "demand": "Demand potential",
    "population": "Population density",
    "income": "Income strength",
    "accessibility": "Road accessibility",
    "competition": "Competition gap",
}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
  --cm-bg: #070d18;
  --cm-panel: rgba(12, 22, 39, .72);
  --cm-border: rgba(148, 163, 184, .17);
  --cm-text: #e6edf7;
  --cm-muted: #93a6bd;
  --cm-cyan: #22d3ee;
  --cm-lime: #a3e635;
  --cm-violet: #a78bfa;
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
  font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.stApp {
  background:
    radial-gradient(1200px 620px at 6% -12%, rgba(34, 211, 238, .105), transparent 62%),
    radial-gradient(900px 500px at 100% -5%, rgba(99, 102, 241, .13), transparent 58%),
    #070d18;
}
header[data-testid="stHeader"] { background: rgba(7, 13, 24, .35); backdrop-filter: blur(10px); }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0a1424 0%, #070d18 100%);
  border-right: 1px solid var(--cm-border);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1560px; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stAppDeployButton"] { display: none !important; }
h1, h2, h3, h4 { font-family: "Space Grotesk", "Inter", sans-serif !important; letter-spacing: -.02em; color: #f1f5fb; }
p, li, label { color: #c6d3e4; }

/* Hero */
.cm-hero {
  display: flex; justify-content: space-between; align-items: flex-start; gap: 24px;
  padding: 26px 30px; border: 1px solid var(--cm-border); border-radius: 20px;
  background: linear-gradient(135deg, rgba(15, 27, 46, .92), rgba(8, 15, 28, .74));
  position: relative; overflow: hidden; box-shadow: 0 22px 60px rgba(0, 0, 0, .28);
}
.cm-hero::before {
  content: ""; position: absolute; inset: 0;
  background: radial-gradient(620px 250px at 12% 0%, rgba(34, 211, 238, .16), transparent 72%);
  pointer-events: none;
}
.cm-hero-left { display: flex; gap: 18px; align-items: flex-start; position: relative; z-index: 1; }
.cm-mark {
  width: 58px; height: 58px; border-radius: 17px; flex: 0 0 auto;
  background: linear-gradient(135deg, #22d3ee, #6366f1);
  display: flex; align-items: center; justify-content: center; font-size: 30px;
  box-shadow: 0 10px 26px rgba(34, 211, 238, .26);
}
.cm-eyebrow {
  font-size: 10px; letter-spacing: .22em; text-transform: uppercase;
  color: #7dd3fc; font-weight: 700; margin-bottom: 7px;
}
.cm-hero h1 { font-size: 39px; line-height: 1.02; margin: 0; color: #f8fafc; font-weight: 700; }
.cm-hero h1 span {
  background: linear-gradient(90deg, #67e8f9, #a78bfa);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.cm-hero p { margin: 10px 0 0; color: #a8bacf; font-size: 13.5px; max-width: 740px; line-height: 1.55; }
.cm-hero-right {
  display: flex; flex-direction: column; align-items: flex-end; gap: 8px;
  position: relative; z-index: 1; padding-top: 2px;
}
.cm-pill {
  display: inline-flex; align-items: center; gap: 6px; padding: 5px 10px;
  border-radius: 999px; border: 1px solid var(--cm-border);
  background: rgba(8, 15, 28, .62); color: #cbd5e1;
  font-size: 10.5px; letter-spacing: .045em; font-weight: 600; white-space: nowrap;
}
.cm-pill.live { color: #86efac; border-color: rgba(134, 239, 172, .28); background: rgba(22, 101, 52, .16); }
.cm-pill.fallback { color: #fcd34d; border-color: rgba(252, 211, 77, .28); background: rgba(120, 53, 15, .18); }
.cm-pill.cyan { color: #67e8f9; border-color: rgba(103, 232, 249, .25); }

/* Metric strip */
.cm-metrics { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin: 16px 0 14px; }
.cm-metric {
  border: 1px solid var(--cm-border); border-top: 2px solid var(--accent);
  border-radius: 13px; padding: 13px 15px 12px;
  background: linear-gradient(145deg, rgba(15, 27, 46, .82), rgba(8, 15, 28, .65));
}
.cm-metric-label { font-size: 9.5px; letter-spacing: .12em; text-transform: uppercase; color: #8fa4bd; font-weight: 700; }
.cm-metric-value { font-family: "Space Grotesk", sans-serif; font-size: 27px; line-height: 1.18; color: #f8fafc; font-weight: 600; margin-top: 4px; }
.cm-metric-sub { font-size: 10.5px; color: #8fa4bd; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* View switcher (segmented control, so only the active view mounts) */
[data-testid="stButtonGroup"] {
  gap: 6px; background: rgba(10, 19, 34, .62); padding: 6px;
  border-radius: 13px; border: 1px solid var(--cm-border); margin-bottom: 4px;
}
[data-testid="stButtonGroup"] [data-testid^="stBaseButton-segmented_control"] {
  border: 0; border-radius: 8px; padding: 8px 15px;
  color: #9fb3cb; font-size: 12.5px; font-weight: 600;
  height: auto; min-height: 0; background: transparent;
}
[data-testid="stButtonGroup"] [data-testid^="stBaseButton-segmented_control"]:hover {
  background: rgba(148, 163, 184, .08); color: #e2e8f0;
}
[data-testid="stButtonGroup"] [data-testid="stBaseButton-segmented_controlActive"] {
  background: rgba(34, 211, 238, .11) !important; color: #e0f2fe !important;
}

/* Section headings */
.cm-section-head { margin: 4px 0 10px; }
.cm-section-head .eyebrow { font-size: 9.5px; letter-spacing: .16em; text-transform: uppercase; color: #67e8f9; font-weight: 700; }
.cm-section-head h2 { font-size: 22px; margin: 3px 0 2px; }
.cm-section-head p { font-size: 12.5px; color: #93a6bd; margin: 0; }

/* Cards / rail */
.cm-panel {
  border: 1px solid var(--cm-border); border-radius: 14px; padding: 16px 18px;
  background: rgba(10, 19, 34, .6); margin-bottom: 12px;
}
.cm-panel h4 { font-size: 14px; margin: 0 0 8px; }
.cm-panel p { font-size: 12px; color: #93a6bd; line-height: 1.55; margin: 0 0 6px; }
.cm-site-card {
  border: 1px solid var(--cm-border); border-left: 3px solid var(--accent);
  border-radius: 12px; padding: 12px 14px 11px; margin-bottom: 10px;
  background: linear-gradient(140deg, rgba(15, 27, 46, .86), rgba(9, 16, 29, .72));
}
.cm-site-top { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.cm-site-rank { font-family: "Space Grotesk", sans-serif; font-size: 12px; color: var(--accent); font-weight: 700; letter-spacing: .03em; }
.cm-tier {
  font-size: 8.5px; letter-spacing: .10em; text-transform: uppercase; font-weight: 800;
  padding: 2px 7px; border-radius: 999px; border: 1px solid currentColor;
}
.cm-site-name { font-size: 13.5px; color: #f1f5f9; font-weight: 600; margin-top: 6px; }
.cm-site-score { font-family: "Space Grotesk", sans-serif; font-size: 25px; font-weight: 700; color: #f8fafc; line-height: 1.1; margin: 2px 0 6px; }
.cm-site-score span { font-size: 11px; color: #7f93ab; font-weight: 500; }
.cm-bar { height: 5px; border-radius: 999px; background: rgba(148, 163, 184, .14); overflow: hidden; }
.cm-bar > i { display: block; height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--accent), #a3e635); }
.cm-site-meta { font-size: 10px; color: #8fa4bd; margin-top: 7px; line-height: 1.5; }

/* Sidebar */
.cm-side-brand { display: flex; align-items: center; gap: 11px; margin-bottom: 18px; }
.cm-side-bolt {
  width: 38px; height: 38px; border-radius: 11px; display: flex; align-items: center; justify-content: center;
  font-size: 20px; background: linear-gradient(135deg, #22d3ee, #6366f1);
  box-shadow: 0 6px 18px rgba(34, 211, 238, .2);
}
.cm-side-title { font-family: "Space Grotesk", sans-serif; font-size: 17px; color: #f1f5f9; font-weight: 700; line-height: 1.1; }
.cm-side-sub { font-size: 8.5px; letter-spacing: .24em; color: #67e8f9; font-weight: 700; }
.cm-side-label { font-size: 9px; letter-spacing: .16em; text-transform: uppercase; color: #7890aa; font-weight: 800; margin: 12px 0 2px; }

/* Charts */
.stBarChart { border: 1px solid var(--cm-border); border-radius: 13px; padding: 10px 12px; background: rgba(10, 19, 34, .5); }

/* Buttons */
.stButton > button, .stDownloadButton > button {
  border-radius: 9px; border: 1px solid rgba(34, 211, 238, .32);
  background: rgba(34, 211, 238, .09); color: #cffafe; font-weight: 600; font-size: 12.5px;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: rgba(34, 211, 238, .6); background: rgba(34, 211, 238, .16); color: #ecfeff;
}

/* Dataframes */
[data-testid="stDataFrame"] { border: 1px solid var(--cm-border); border-radius: 12px; overflow: hidden; }

/* Method */
.cm-step {
  border: 1px solid var(--cm-border); border-radius: 13px; padding: 14px 16px;
  background: rgba(10, 19, 34, .55); height: 100%;
}
.cm-step .num { font-family: "Space Grotesk", sans-serif; font-size: 11px; color: #67e8f9; font-weight: 700; letter-spacing: .1em; }
.cm-step h4 { font-size: 13.5px; margin: 4px 0 5px; }
.cm-step p { font-size: 11.5px; color: #93a6bd; line-height: 1.5; margin: 0; }
.cm-formula {
  font-family: "SFMono-Regular", Consolas, monospace; font-size: 12px; color: #c4f1ff;
  background: rgba(8, 15, 28, .8); border: 1px solid var(--cm-border);
  border-radius: 9px; padding: 10px 12px; margin: 8px 0; overflow-x: auto;
}
.cm-footer { margin-top: 26px; padding-top: 14px; border-top: 1px solid var(--cm-border); font-size: 10.5px; color: #6f849c; text-align: center; }
.cm-note { font-size: 10.5px; color: #8fa4bd; line-height: 1.5; }

@media (max-width: 1100px) {
  .cm-metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .cm-hero { flex-direction: column; }
  .cm-hero-right { flex-direction: row; flex-wrap: wrap; align-items: flex-start; }
}
@media (max-width: 900px) {
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
  [data-testid="stColumn"] { width: 100% !important; flex: 1 1 100% !important; min-width: 0 !important; }
}
</style>
"""


def escape(value: object) -> str:
    return html.escape(str(value))


def metric_card_html(label: str, value: str, sub: str, accent: str = "#22d3ee") -> str:
    # Single line on purpose: a leading-indented multi-line block would be parsed
    # as a markdown code block instead of five sibling cards.
    return (
        f'<div class="cm-metric" style="--accent:{accent};">'
        f'<div class="cm-metric-label">{escape(label)}</div>'
        f'<div class="cm-metric-value">{escape(value)}</div>'
        f'<div class="cm-metric-sub">{escape(sub)}</div>'
        f"</div>"
    )


def render_site_card(row: pd.Series) -> None:
    accent = TIER_ACCENTS.get(str(row["tier"]), "#22d3ee")
    score = float(row["opportunity_score"])
    st.markdown(
        f"""
        <div class="cm-site-card" style="--accent:{accent};">
          <div class="cm-site-top">
            <span class="cm-site-rank">#{int(row['rank'])} · CANDIDATE</span>
            <span class="cm-tier" style="color:{accent};">{escape(row['tier'])}</span>
          </div>
          <div class="cm-site-name">{escape(row['zone'])}</div>
          <div class="cm-site-score">{score:.1f}<span>/100</span></div>
          <div class="cm-bar"><i style="width:{max(2.0, min(score, 100.0)):.1f}%"></i></div>
          <div class="cm-site-meta">
            Demand {float(row['demand_score']):.0f} · Access {float(row['accessibility_score']):.0f} · Gap {float(row['competition_gap_score']):.0f}<br>
            {int(row['nearest_ev_m']):,} m to nearest charger · {int(row['ev_within_radius'])} within 1.6 km
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_head(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="cm-section-head">
          <div class="eyebrow">{escape(eyebrow)}</div>
          <h2>{escape(title)}</h2>
          <p>{escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Sidebar — controls
# --------------------------------------------------------------------------- #

st.sidebar.markdown(
    """
    <div class="cm-side-brand">
      <div class="cm-side-bolt">⚡</div>
      <div>
        <div class="cm-side-title">ChargeMap</div>
        <div class="cm-side-sub">INTELLIGENCE</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.caption("EV charging site planner · Austin, TX")

st.sidebar.markdown('<div class="cm-side-label">01 · Data source</div>', unsafe_allow_html=True)
source_label = st.sidebar.selectbox(
    "Source",
    options=list(DATA_SOURCES.keys()),
    label_visibility="collapsed",
    help="OpenStreetMap fetches live POIs and major roads; simulated data is deterministic and needs no network.",
)
source_mode = DATA_SOURCES[source_label]

if st.sidebar.button("↻ Refresh OSM cache", width="stretch", help="Clear the local Overpass cache and reload."):
    for cache_file in CACHE_DIR.glob("osm_*.json"):
        cache_file.unlink(missing_ok=True)
    st.cache_data.clear()
    st.session_state["refresh_nonce"] = st.session_state.get("refresh_nonce", 0) + 1
    st.rerun()

st.sidebar.markdown('<div class="cm-side-label">02 · Demand model</div>', unsafe_allow_html=True)
eps_m = st.sidebar.slider(
    "DBSCAN radius (m)",
    min_value=250,
    max_value=1000,
    value=550,
    step=50,
    help="Neighbourhood radius used to group POIs into a demand zone.",
)
min_samples = st.sidebar.slider(
    "Minimum POIs per zone",
    min_value=3,
    max_value=12,
    value=5,
    step=1,
    help="DBSCAN core-point threshold; higher values yield fewer, denser zones.",
)

st.sidebar.markdown('<div class="cm-side-label">03 · Scoring weights</div>', unsafe_allow_html=True)
weight_inputs: dict[str, float] = {}
for key, label in WEIGHT_LABELS.items():
    weight_inputs[key] = float(
        st.sidebar.slider(
            label,
            min_value=0,
            max_value=60,
            value=int(round(DEFAULT_WEIGHTS[key] * 100)),
            step=5,
            help=f"Relative importance of {label.lower()} in the composite score.",
        )
    )

effective_weights = normalise_weights(weight_inputs)
st.sidebar.caption(
    "Effective weights (auto-normalised): "
    + " · ".join(f"{WEIGHT_LABELS[key].split()[0]} {value * 100:.0f}%" for key, value in effective_weights.items())
)

st.sidebar.markdown('<div class="cm-side-label">04 · Shortlist</div>', unsafe_allow_html=True)
n_sites = st.sidebar.slider("Candidates to shortlist", 3, 8, 5, 1)
min_spacing_m = st.sidebar.slider(
    "Minimum site spacing (m)",
    min_value=600,
    max_value=3000,
    value=1200,
    step=100,
    help="Spatial suppression: selected sites must be at least this far apart.",
)
max_per_zone = st.sidebar.slider(
    "Max sites per district",
    min_value=1,
    max_value=4,
    value=2,
    step=1,
    help="Keeps the shortlist from stacking every candidate inside the single hottest DBSCAN zone.",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div class="cm-note">
      <b style="color:#cbd5e1;">Prototype data.</b> POIs and major roads come from
      OpenStreetMap when reachable. Demographics and incumbent chargers are simulated
      for planning exercises and are not real market data.
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Data + model
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def cached_load(source: str, seed: int = 42, refresh_nonce: int = 0):
    del refresh_nonce  # cache-busting input only
    return load_all_data(AUSTIN_BBOX, source=source, seed=seed, use_cache=True)


with st.spinner("Loading Austin layers…"):
    bundle = cached_load(source_mode, 42, st.session_state.get("refresh_nonce", 0))

pois = bundle["pois"]
roads = bundle["roads"]
ev_stations = bundle["ev_stations"]
grid = bundle["grid"]
meta = bundle["meta"]

with st.spinner("Clustering demand and scoring candidate cells…"):
    clusters = cluster_demand_zones(pois, eps_m=eps_m, min_samples=min_samples)
    scored = score_grid(grid, pois, roads, ev_stations, clusters, weights=effective_weights)
    shortlist = shortlist_sites(
        scored,
        clusters,
        n_sites=n_sites,
        min_spacing_m=min_spacing_m,
        max_per_zone=max_per_zone,
    )

n_clusters = int(clusters.attrs.get("n_clusters", len(clusters)))
n_noise = int(clusters.attrs.get("n_noise", 0))
best_score = float(scored["opportunity_score"].max()) if not scored.empty else 0.0
best_zone = str(shortlist.iloc[0]["zone"]) if not shortlist.empty else "No cluster"
status_text = str(meta.get("status", "Unknown"))
is_fallback = bool(meta.get("fallback", False))
if source_mode == "simulated":
    status_variant = "cyan"
elif is_fallback:
    status_variant = "fallback"
else:
    status_variant = "live"

st.markdown(
    f"""
    <div class="cm-hero">
      <div class="cm-hero-left">
        <div class="cm-mark">⚡</div>
        <div>
          <div class="cm-eyebrow">EV charging · location intelligence</div>
          <h1>ChargeMap <span>Intelligence</span></h1>
          <p>
            A spatial decision studio for the next public charging site in {escape(CITY_NAME)}.
            DBSCAN finds the demand zones, a five-criteria model scores every grid cell, and a
            spatially suppressed shortlist turns the heat map into an action list.
          </p>
        </div>
      </div>
      <div class="cm-hero-right">
        <span class="cm-pill {status_variant}">● {escape(status_text)}</span>
        <span class="cm-pill">Austin, TX</span>
        <span class="cm-pill">DBSCAN + MCDA</span>
        <span class="cm-pill">UTM 14N</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

source_note = "Simulated source" if source_mode == "simulated" else ("Simulated fallback" if is_fallback else "OpenStreetMap")
st.markdown(
    '<div class="cm-metrics">'
    + metric_card_html("Commercial POIs", f"{len(pois):,}", source_note, "#22d3ee")
    + metric_card_html("Demand zones", f"{n_clusters}", f"{n_noise} noise points · ε {eps_m} m", "#a78bfa")
    + metric_card_html("Existing chargers", f"{len(ev_stations)}", "Simulated competitors", "#fb7185")
    + metric_card_html("Best cell score", f"{best_score:.1f}", best_zone, "#facc15")
    + metric_card_html(
        "Shortlist",
        f"{len(shortlist)}",
        f"{min_spacing_m / 1000:.1f} km spacing · ≤{max_per_zone}/district",
        "#a3e635",
    )
    + "</div>",
    unsafe_allow_html=True,
)

VIEW_OPTIONS = ["◈ Opportunity map", "◉ Demand zones", "▤ Site shortlist", "▦ Dataset", "⌘ Method & data"]
active_view = st.segmented_control(
    "Dashboard view",
    options=VIEW_OPTIONS,
    default=VIEW_OPTIONS[0],
    label_visibility="collapsed",
    key="active_view",
    width="stretch",
)
if not active_view:
    active_view = VIEW_OPTIONS[0]

# --------------------------------------------------------------------------- #
# View 1 — Opportunity map
# --------------------------------------------------------------------------- #


def render_overview() -> None:
    section_head(
        "Decision surface",
        "Where the next charger earns its keep",
        "Coloured cells are MinMax-normalised opportunity scores. Numbered pins are the shortlist; dashed rings show 900 m coverage.",
    )
    map_column, rail_column = st.columns([2.55, 1.0], gap="large")

    with map_column:
        opportunity_map = create_opportunity_map(
            scored,
            shortlist,
            ev_stations,
            clusters,
            bbox=AUSTIN_BBOX,
            coverage_radius_m=900,
        )
        st_folium(
            opportunity_map,
            height=618,
            use_container_width=True,
            returned_objects=[],
            key="opportunity_map",
        )

    with rail_column:
        st.markdown(
            """
            <div class="cm-section-head" style="margin-top:2px;">
              <div class="eyebrow">Shortlist snapshot</div>
              <h2 style="font-size:18px;">Top candidates</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if shortlist.empty:
            st.warning("No candidate cells passed the spacing constraint. Reduce spacing or increase the search grid.")
        else:
            for _, site in shortlist.head(4).iterrows():
                render_site_card(site)

        top_zone_detail = ""
        if not shortlist.empty:
            top_row = shortlist.iloc[0]
            top_zone_detail = (
                f"<p><b style='color:#dbeafe;'>#{int(top_row['rank'])} {escape(top_row['zone'])}</b> "
                f"leads with {float(top_row['opportunity_score']):.1f}/100. It combines "
                f"{float(top_row['demand_score']):.0f} demand, {float(top_row['population_score']):.0f} population, "
                f"{float(top_row['income_score']):.0f} income, {float(top_row['accessibility_score']):.0f} access, "
                f"and a {float(top_row['competition_gap_score']):.0f} competition gap.</p>"
            )
        st.markdown(
            f"""
            <div class="cm-panel">
              <h4>Reading the map</h4>
              {top_zone_detail}
              <p>Pins are chosen greedily from the highest-scoring cells, skipping any cell within the spacing radius of an earlier pick and capping sites per district.</p>
              <p style="margin-bottom:0;">Tune the weights in the sidebar; all criteria re-normalise instantly.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# --------------------------------------------------------------------------- #
# View 2 — Demand zones
# --------------------------------------------------------------------------- #

def render_demand() -> None:
    section_head(
        "DBSCAN demand discovery",
        f"{n_clusters} commercial hotspots, {n_noise} scattered POIs",
        f"Density-based clustering with ε = {eps_m} m and min_samples = {min_samples}. Zone demand weights malls above parking.",
    )
    cluster_map = create_cluster_map(pois, clusters, ev_stations, bbox=AUSTIN_BBOX)
    st_folium(cluster_map, height=510, use_container_width=True, returned_objects=[], key="cluster_map")

    table_column, chart_column = st.columns([1.35, 1.0], gap="large")
    with table_column:
        st.markdown("#### Zone ranking")
        if clusters.empty:
            st.info("No DBSCAN zones formed with these parameters. Try a larger radius or fewer minimum POIs.")
        else:
            zone_table = pd.DataFrame(
                {
                    "Zone": clusters["name"],
                    "POIs": clusters["poi_count"],
                    "Demand": clusters["demand_index"],
                    "Dominant POI": clusters["dominant_category"].str.replace("_", " ").str.title(),
                    "Radius (m)": clusters["radius_m"],
                }
            )
            st.dataframe(
                zone_table,
                width="stretch",
                hide_index=True,
                height=336,
                column_config={
                    "Demand": st.column_config.ProgressColumn(
                        "Demand index", min_value=0, max_value=100, format="%.0f"
                    ),
                    "POIs": st.column_config.NumberColumn("POIs", format="%d"),
                    "Radius (m)": st.column_config.NumberColumn("Radius (m)", format="%d"),
                },
            )
            st.caption(
                "Demand index = log-scaled, share-weighted POI mix (mall 3.0, marketplace 2.4, commercial 1.9, parking 1.0), scaled 0–100."
            )
    with chart_column:
        st.markdown("#### Demand by zone")
        if not clusters.empty:
            chart_data = (
                clusters.set_index("name")["demand_index"].sort_values(ascending=True).tail(10)
            )
            st.bar_chart(chart_data, horizontal=True, height=336, color="#22d3ee")
            st.caption("Top 10 zones by demand index. Hover the map centres for POI counts and dominant POI type.")

# --------------------------------------------------------------------------- #
# View 3 — Shortlist
# --------------------------------------------------------------------------- #

def render_shortlist() -> None:
    section_head(
        "Action list",
        "The candidate sites, ranked",
        "Greedy maximum-coverage selection respects both the minimum spacing and the district cap, so the shortlist reads as distinct trade areas.",
    )
    download_column, note_column = st.columns([1, 2.4], gap="large")
    with download_column:
        if not shortlist.empty:
            export_columns = [
                "rank",
                "site_name",
                "zone",
                "opportunity_score",
                "tier",
                "demand_score",
                "population_score",
                "income_score",
                "accessibility_score",
                "competition_gap_score",
                "nearest_ev_m",
                "ev_within_radius",
                "poi_within_radius",
                "lat",
                "lon",
            ]
            csv = shortlist[[column for column in export_columns if column in shortlist.columns]].to_csv(index=False)
            st.download_button(
                "↓ Download shortlist CSV",
                data=csv,
                file_name="chargemap_austin_shortlist.csv",
                mime="text/csv",
                width="stretch",
            )
    with note_column:
        st.markdown(
            f"""
            <div class="cm-note" style="padding-top:8px;">
              Each site is a ~550 m demographic grid cell. Scores are relative to this study area:
              <b>Prime</b> = top decile, <b>Strong</b> = next 20%, <b>Moderate</b> = next 30%.
              Coverage rings are 900 m and do not model road barriers.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if shortlist.empty:
        st.warning("Nothing to show. Relax the spacing rule or adjust the DBSCAN parameters.")
    else:
        shortlist_table = pd.DataFrame(
            {
                "Rank": shortlist["rank"],
                "Site": shortlist["site_name"],
                "Score": shortlist["opportunity_score"],
                "Tier": shortlist["tier"],
                "Demand": shortlist["demand_score"],
                "Population": shortlist["population_score"],
                "Income": shortlist["income_score"],
                "Access": shortlist["accessibility_score"],
                "Gap": shortlist["competition_gap_score"],
                "Nearest EV (m)": shortlist["nearest_ev_m"],
            }
        )
        st.dataframe(
            shortlist_table,
            width="stretch",
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn("#", format="%d", width="small"),
                "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
                "Demand": st.column_config.NumberColumn("Demand", format="%.0f"),
                "Population": st.column_config.NumberColumn("Population", format="%.0f"),
                "Income": st.column_config.NumberColumn("Income", format="%.0f"),
                "Access": st.column_config.NumberColumn("Access", format="%.0f"),
                "Gap": st.column_config.NumberColumn("Gap", format="%.0f"),
                "Nearest EV (m)": st.column_config.NumberColumn("Nearest EV (m)", format="%d"),
            },
        )

        chart_column, detail_column = st.columns([1.45, 1.0], gap="large")
        with chart_column:
            st.markdown("#### Criterion scores by candidate")
            breakdown = criterion_breakdown(shortlist, effective_weights)
            pivot = breakdown.pivot(index="site_name", columns="label", values="score")
            st.bar_chart(pivot, height=330)
            st.caption("Unweighted criterion scores (0–100) before applying the sidebar weights.")
        with detail_column:
            st.markdown("#### Score composition")
            focused_name = st.selectbox(
                "Candidate",
                options=list(shortlist["site_name"]),
                label_visibility="collapsed",
            )
            focused = breakdown[breakdown["site_name"] == focused_name]
            for _, criterion in focused.iterrows():
                accent = TIER_ACCENTS.get(str(shortlist.iloc[0]["tier"]), "#22d3ee")
                st.markdown(
                    f"""
                    <div style="margin:0 0 9px;">
                      <div style="display:flex;justify-content:space-between;gap:8px;font-size:11px;color:#a9bbd1;">
                        <span>{escape(criterion['label'])} <span style="color:#64748b;">· w {criterion['weight'] * 100:.0f}%</span></span>
                        <b style="color:#e2e8f0;">{float(criterion['score']):.0f}</b>
                      </div>
                      <div class="cm-bar" style="margin-top:4px;"><i style="width:{max(2.0, min(float(criterion['score']), 100.0)):.1f}%; background:linear-gradient(90deg,{accent},#a3e635);"></i></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            total = float(focused["weighted_points"].sum())
            st.caption(f"Weighted contribution: {total:.1f} of 100 potential points. Remaining score comes from relative normalisation.")

# --------------------------------------------------------------------------- #
# View 4 — Dataset
# --------------------------------------------------------------------------- #

def render_dataset() -> None:
    manifest = meta.get("dataset_manifest", {})
    section_head(
        "Teacher-ready evidence",
        "Dataset saved before prediction",
        "These are the exact raw layers written to disk before DBSCAN clustering and opportunity scoring. Use the previews or download the CSV files for review.",
    )
    saved_at = manifest.get("created_at", meta.get("loaded_at", ""))
    dataset_dir = meta.get("dataset_dir", "data/chargemap_dataset")
    st.success(f"Dataset snapshot ready · {saved_at} · stored in `{dataset_dir}`")

    frames = {
        "Commercial POIs": pois,
        "Major roads": roads,
        "Existing chargers": ev_stations,
        "Demographic grid": grid,
    }
    summary = pd.DataFrame(
        {
            "Layer": list(frames),
            "Rows": [len(frame) for frame in frames.values()],
            "Columns": [len(frame.drop(columns=["geometry"], errors="ignore").columns) for frame in frames.values()],
            "Saved file": [manifest.get("files", {}).get(key, f"{key.lower().replace(' ', '_')}.csv") for key in ("pois", "roads", "ev_stations", "demographic_grid")],
        }
    )
    st.dataframe(summary, width="stretch", hide_index=True)

    preview_column, download_column = st.columns([1.7, 1.0], gap="large")
    with preview_column:
        selected_layer = st.selectbox("Preview layer", options=list(frames), label_visibility="collapsed")
        preview = frames[selected_layer].drop(columns=["geometry"], errors="ignore")
        st.dataframe(preview.head(100), width="stretch", height=370, hide_index=True)
        st.caption(f"Showing the first {min(100, len(preview))} of {len(preview):,} rows.")
    with download_column:
        st.markdown("#### Download raw layers")
        for label, frame in frames.items():
            file_key = {
                "Commercial POIs": "pois",
                "Major roads": "roads",
                "Existing chargers": "ev_stations",
                "Demographic grid": "demographic_grid",
            }[label]
            csv_data = frame.drop(columns=["geometry"], errors="ignore").to_csv(index=False)
            st.download_button(
                f"↓ {label} CSV",
                data=csv_data,
                file_name=f"{file_key}.csv",
                mime="text/csv",
                width="stretch",
                key=f"download_{file_key}",
            )
        st.markdown(
            '<div class="cm-panel"><h4>Reproducibility note</h4>'
            '<p>The snapshot is regenerated on each data load. The manifest records the source, timestamp, row counts, and the order in which the model consumes the files.</p></div>',
            unsafe_allow_html=True,
        )

# --------------------------------------------------------------------------- #
# View 5 — Method & data
# --------------------------------------------------------------------------- #

def render_method() -> None:
    section_head(
        "Transparent by construction",
        "How ChargeMap scores a site",
        "Every number in the dashboard traces back to one of these four stages. Nothing is a black box.",
    )

    step_columns = st.columns(4, gap="medium")
    steps = [
        ("01 · ACQUIRE", "OSM + simulated layers", "Overpass returns commercial POIs and major-road access points. Existing chargers and demographic income/population grids are deterministic simulations for Austin."),
        ("02 · CLUSTER", "DBSCAN demand zones", "POIs are projected to UTM 14N and clustered in metres. Each zone gets a demand index from its POI mix, then spreads demand with a Gaussian kernel."),
        ("03 · SCORE", "Five-criteria MCDA", "MinMaxScaler puts demand, population, income, accessibility and competition gap on a common 0–1 scale before applying the sidebar weights."),
        ("04 · SHORTLIST", "Greedy spatial suppression", "Cells are sorted by score and accepted only if they respect both the chosen spacing and the per-district cap — distinct trade areas, not adjacent noise."),
    ]
    for column, (eyebrow, title, body) in zip(step_columns, steps):
        with column:
            st.markdown(
                f"""
                <div class="cm-step">
                  <div class="num">{escape(eyebrow)}</div>
                  <h4>{escape(title)}</h4>
                  <p>{escape(body)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("#### The scoring equation")
    method_columns = st.columns([1.4, 1.0], gap="large")
    with method_columns[0]:
        weight_terms = " + ".join(
            f"{effective_weights[key]:.2f}·x̃<sub>{key}</sub>" for key in DEFAULT_WEIGHTS
        )
        st.markdown(
            f"""
            <div class="cm-formula">
              score<sub>i</sub> = 100 × ( {weight_terms} )
            </div>
            <div class="cm-note">
              x̃ is each criterion's MinMaxScaler-normalised value across all grid cells.
              Weights are re-normalised to sum to 1.0 after the sidebar sliders change.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("#### Current weights")
        weight_table = pd.DataFrame(
            {
                "Criterion": [CRITERIA_LABELS[key] for key in DEFAULT_WEIGHTS],
                "Weight (%)": [effective_weights[key] * 100 for key in DEFAULT_WEIGHTS],
                "What it captures": [
                    "DBSCAN cluster gravity plus POI density near the cell.",
                    "Simulated residents per km² — charging opportunity volume.",
                    "Simulated purchasing power / EV adoption headroom.",
                    "Closer major-road access points score higher, with exponential decay.",
                    "Distance from incumbents (65%) plus scarcity of stations within 1.6 km (35%).",
                ],
            }
        )
        st.dataframe(
            weight_table,
            width="stretch",
            hide_index=True,
            column_config={
                "Weight (%)": st.column_config.ProgressColumn("Weight", min_value=0, max_value=100, format="%.0f"),
            },
        )
    with method_columns[1]:
        st.markdown("#### Data lineage")
        source_table = pd.DataFrame(
            {
                "Layer": ["Commercial POIs", "Major roads", "Demographics", "Competitors", "Demand zones"],
                "Source": [
                    "OpenStreetMap" if not is_fallback else "Simulated",
                    "OpenStreetMap" if not is_fallback else "Simulated",
                    "Simulated",
                    "Simulated",
                    "DBSCAN",
                ],
                "Rows": [len(pois), len(roads), len(grid), len(ev_stations), n_clusters],
            }
        )
        st.dataframe(source_table, width="stretch", hide_index=True)
        st.markdown(
            f"""
            <div class="cm-panel">
              <h4>Assumptions & limits</h4>
              <p>Demographics are synthetic surfaces, not census tract data. Competitor stations are randomized around active districts.</p>
              <p>The model ignores grid capacity, rent, zoning, traffic counts, highway barriers and precise walkability — validate a shortlisted parcel before committing capital.</p>
              <p style="margin-bottom:0;">OSM completeness varies. Batched mode reflects a snapshot rather than a live operations feed.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Inspect the raw acquisition layers"):
        show_raw = st.toggle("Show raw data map", value=False, key="show_raw_map")
        if show_raw:
            source_map = create_source_map(pois, roads, ev_stations, bbox=AUSTIN_BBOX)
            st_folium(source_map, height=440, use_container_width=True, returned_objects=[], key="source_map")
        st.caption(f"Loaded {meta.get('loaded_at', '')} · {status_text} · bbox {AUSTIN_BBOX}")

if active_view == VIEW_OPTIONS[0]:
    render_overview()
elif active_view == VIEW_OPTIONS[1]:
    render_demand()
elif active_view == VIEW_OPTIONS[2]:
    render_shortlist()
elif active_view == VIEW_OPTIONS[3]:
    render_dataset()
else:
    render_method()

st.markdown(
    """
    <div class="cm-footer">
      ChargeMap Intelligence · Streamlit · GeoPandas · scikit-learn DBSCAN · Folium &nbsp;|&nbsp;
      Prototype for planning exploration — demographics and competitor chargers are simulated.
    </div>
    """,
    unsafe_allow_html=True,
)
