# ⚡ ChargeMap Intelligence

An EV charging **business location planner** for Austin, Texas. ChargeMap combines
commercial activity, simulated demographics, existing-charger competition and road
access into a ranked shortlist of candidate charging sites.

The app is a modular Streamlit dashboard built with GeoPandas, scikit-learn
(DBSCAN + MinMaxScaler), Folium and the OpenStreetMap Overpass API.

> **Prototype data:** commercial POIs and major-road access points come from
> OpenStreetMap. Existing chargers, population density and income are simulated for
> planning exercises — they are **not** market research or a substitute for parcel-level
> due diligence.

---

## Contents

- [What it does](#what-it-does)
- [Features](#features)
- [Tech stack](#tech-stack)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Component reference](#component-reference)
- [How the model works](#how-the-model-works)
- [Using the dashboard](#using-the-dashboard)
- [Setup and run](#setup-and-run)
- [Configuration](#configuration)
- [Data, caching and offline behavior](#data-caching-and-offline-behavior)
- [Troubleshooting](#troubleshooting)
- [Verification](#verification)
- [Limitations and next steps](#limitations-and-next-steps)
- [Data credits](#data-credits)

---

## What it does

Given a central-Austin study area, ChargeMap answers three planning questions:

1. **Where is commercial demand concentrated?** POIs are grouped into demand zones
   with DBSCAN.
2. **Which grid cells make the strongest charging sites?** Every cell is scored on
   five weighted criteria.
3. **What should I actually visit first?** A greedy, spatially suppressed shortlist
   returns distinct candidate trade areas rather than five adjacent cells.

The result is an interactive map, a zone ranking, a scored candidate table and a
downloadable CSV.

---

## Features

- **OpenStreetMap acquisition** via the Overpass API (`overpy`, with a `requests`
  fallback and mirror endpoints).
- **24-hour disk cache** so repeat dashboard loads are fast.
- **Deterministic simulated fallback** for POIs and roads when OSM is unavailable;
  the app always renders.
- **DBSCAN demand zones** projected in metres (UTM zone 14N), with neighborhood
  labels, demand indices and convex-hull polygons.
- **Five-criteria MCDA model** using `MinMaxScaler` and user-adjustable weights.
- **Spatially constrained shortlist** with minimum spacing and a per-district cap.
- **Interactive Folium maps**: opportunity heat surface, candidate pins with 900 m
  coverage rings, DBSCAN hulls, competitor chargers and raw acquisition layers.
- **CSV export** of the shortlist.
- **Responsive dark UI** with a sidebar control panel and four switchable views.

---

## Tech stack

| Layer | Library / tool | Version |
| --- | --- | --- |
| App framework | Streamlit | 1.49.1 |
| Map components | streamlit-folium | 0.25.1 |
| Mapping | Folium / Branca | 0.20.0 / 0.8.2 |
| Tabular data | pandas | 2.2.3 |
| Geospatial data | GeoPandas | 1.0.1 |
| Geometry | Shapely | 2.0.7 |
| Numerics | NumPy | 2.0.2 |
| Clustering / scaling | scikit-learn | 1.6.1 |
| HTTP | requests | 2.32.5 |
| Overpass client | overpy | 0.7 |

Python **3.9+** is required. The project was built and verified with Python 3.9.6.

---

## Architecture

```text
OpenStreetMap Overpass API
        │  (overpy → requests fallback)
        ▼
data_loader.py ──────────────► POIs, road access points (cached 24 h)
        │
        ├── simulated EV chargers (competition)
        ├── simulated demographic grid (population + income)
        └── study-area bounding box
        ▼
model.py
        ├── DBSCAN demand zones
        ├── MinMaxScaler + weighted criteria
        └── greedy spacing / district-cap shortlist
        ▼
app.py ──────► sidebar controls, metrics, tables, CSV export
        │
        └── map_viz.py ──────► Folium opportunity / zone / raw-layer maps
```

Only the **active dashboard view** is computed and mounted at a time. This keeps
the app responsive and avoids Streamlit sizing issues for maps and dataframes inside
hidden tab panels.

---

## Project structure

```text
.
├── app.py                    # Streamlit entry point, theme, sidebar, views
├── data_loader.py            # OSM + simulated data acquisition
├── model.py                  # DBSCAN, MCDA scoring, shortlist selection
├── map_viz.py                # Folium map builders
├── requirements.txt          # Pinned Python dependencies
├── README.md
├── .streamlit/
│   └── config.toml           # Dark theme and headless server defaults
├── .cache/                   # Runtime OSM cache (gitignored)
└── .venv/                    # Local virtual environment (gitignored)
```

---

## Component reference

### `app.py` — dashboard entry point

Run this file with Streamlit. It is responsible for:

- **Page configuration** — wide layout, ⚡ favicon, expanded sidebar.
- **Theme** — custom dark CSS for the hero, metric cards, sidebars, tables and the
  view switcher.
- **Sidebar controls**
  - Data source: `OpenStreetMap · auto fallback` or `Simulated · offline`
  - Refresh OSM cache
  - DBSCAN radius and minimum POIs per zone
  - Five scoring weights
  - Shortlist size, minimum site spacing, max sites per district
- **Data loading** — `load_all_data()` wrapped in `st.cache_data` (12-hour TTL).
- **Model orchestration** — calls `cluster_demand_zones()`, `score_grid()` and
  `shortlist_sites()` on every rerun.
- **Four views**, selected with a single choice control:
  1. **Opportunity map** — heat surface, shortlisted pins, coverage rings and a
     shortlist snapshot rail.
  2. **Demand zones** — DBSCAN hull map, zone ranking table and demand chart.
  3. **Site shortlist** — ranked table, criterion comparison chart, per-candidate
     score bars and CSV download.
  4. **Method & data** — scoring equation, effective weights, lineage table,
     assumptions and an optional raw-layer map.
- **Footer and data warnings** — makes the simulated nature of demographics and
  competitors explicit.

### `data_loader.py` — data acquisition

Key constants:

| Constant | Meaning |
| --- | --- |
| `AUSTIN_BBOX` | Study area `(south, west, north, east)` = `(30.18, -97.83, 30.38, -97.66)` |
| `AUSTIN_CENTER` | Downtown reference point |
| `GRID_STEP_DEG` | Demographic grid spacing, `0.005°` (~550 m) |
| `CACHE_DIR` | `.cache/` beside the source files |
| `OSM_CACHE_TTL_SECONDS` | 24-hour OSM cache lifetime |
| `POI_WEIGHTS` | Demand weights: mall 3.0, marketplace 2.4, commercial 1.9, parking 1.0 |

Main functions:

| Function | Purpose |
| --- | --- |
| `load_all_data(bbox, source, seed, use_cache)` | Returns a bundle with `pois`, `roads`, `ev_stations`, `grid`, `study_area` and `meta` |
| `fetch_osm_features(bbox, timeout, use_cache)` | Fetches POIs + major-road points; tries cache, then `overpy`, then `requests`, then mirrors |
| `generate_mock_pois()` | Deterministic commercial POIs around Austin districts |
| `generate_mock_roads()` | Densified points along major arterials |
| `generate_mock_ev_stations()` | Simulated incumbent chargers with operators, power and stall counts |
| `generate_demographic_grid()` | Gridded simulated population and income surfaces |
| `bbox_as_gdf()` | Study-area polygon for mapping |

The Overpass query requests:

- `amenity=parking`
- `shop=mall`
- `amenity=marketplace`
- `landuse=commercial`
- major-road junctions and traffic signals
- `motorway`, `trunk`, `primary`, `secondary` ways, represented by their centres

All returned frames use WGS84 (`EPSG:4326`) and expose `lat` / `lon` helper columns.

### `model.py` — clustering and scoring

Key constants:

| Constant | Meaning |
| --- | --- |
| `METRIC_CRS` | `EPSG:32614` (UTM zone 14N, metres) |
| `DEFAULT_WEIGHTS` | demand 0.30, population 0.20, income 0.15, accessibility 0.15, competition 0.20 |

Main functions:

| Function | Purpose |
| --- | --- |
| `normalise_weights()` | Clamps weights, falls back to defaults, renormalises to sum 1.0 |
| `cluster_demand_zones()` | DBSCAN over POIs in metres; adds names, demand index, radius and hulls |
| `score_grid()` | Computes five normalized criteria and the weighted opportunity score |
| `shortlist_sites()` | Greedy selection with minimum spacing and a per-zone cap |
| `criterion_breakdown()` | Long-form weighted contributions for charts |

### `map_viz.py` — maps

| Function | Purpose |
| --- | --- |
| `create_opportunity_map()` | Scored grid surface, shortlisted candidate pins, coverage rings, zones and competitors |
| `create_cluster_map()` | DBSCAN hulls, POI members, zone centres and competitor chargers |
| `create_source_map()` | Raw POIs, road points and chargers for data inspection |

Maps use CARTO dark basemap tiles, a score colour ramp, tier-based opacity, candidate
marker pins, legends and OpenStreetMap/CARTO attribution. The initial fit is capped at
zoom 13 so maps render at a useful neighbourhood scale.

### `.streamlit/config.toml`

Sets the dark theme, cyan primary colour, headless server mode and disables usage
statistics.

### `requirements.txt`

Pinned dependencies for reproducible installs. No API keys or external credentials are
needed.

---

## How the model works

### 1. Demand zones with DBSCAN

POIs are projected to UTM 14N, then clustered with:

```python
DBSCAN(eps=550, min_samples=5)
```

`eps` is in metres; both values are adjustable in the sidebar. Each cluster receives:

- **Demand weight** from its POI mix:
  `mall 3.0`, `marketplace 2.4`, `commercial_zone 1.9`, `parking 1.0`
- **Demand index** — log-compressed and scaled to 0–100, so one very dense downtown
  cluster does not flatten every other district.
- **Name** — nearest Austin neighborhood within 4.5 km, otherwise a compass-based
  corridor name.
- **Radius** — 90th-percentile member distance + 120 m, minimum 250 m.
- **Hull** — buffered convex hull of member POIs.

### 2. Five scoring criteria

Every grid cell gets a 0–100 score for:

| Criterion | Sidebar label | Default weight | How it is derived |
| --- | --- | --- | --- |
| Demand potential | Demand potential | 30% | 65% Gaussian demand field from DBSCAN zones + 35% POI count within 850 m |
| Population density | Population density | 20% | Simulated population score for the cell |
| Income strength | Income strength | 15% | Simulated purchasing-power score (EV adoption headroom proxy) |
| Road accessibility | Road accessibility | 15% | Exponential decay of distance to the nearest major-road access point (`exp(-d/700 m)`) |
| Competition gap | Competition gap | 20% | 65% normalized distance to the nearest existing charger + 35% scarcity of chargers within 1.6 km |

All five criteria are normalized together with scikit-learn's `MinMaxScaler`, then
combined:

```text
score_i = 100 × Σ (w_c × x̃_i,c)
```

`x̃` is the MinMax-normalized value across all cells; `w` is renormalized to sum to 1.
Scores are therefore **relative to this study area**, not absolute market potential.

Tiers are percentile-based:

- **Prime** — top 10%
- **Strong** — next 20%
- **Moderate** — next 30%
- **Low** — remaining 40%

### 3. Shortlist selection

Cells are sorted by score. A candidate is accepted only if:

1. It is at least **`min_spacing_m`** (default 1.2 km) from every site already chosen.
2. Its district has fewer than **`max_per_zone`** sites (default 2).

Districts come from the nearest DBSCAN zone; cells farther than 4 km are labeled
`Underserved corridor` and do not consume a district slot. The result is a ranked set
of distinct trade areas, with `site_name`, `zone`, scores, nearest-charger distance
and other fields available for CSV export.

---

## Using the dashboard

### Sidebar

| Section | Control | Effect |
| --- | --- | --- |
| 01 · Data source | Source selector | Load cached/live OSM or fully simulated data |
| 01 · Data source | Refresh OSM cache | Deletes cached OSM files and forces a refetch |
| 02 · Demand model | DBSCAN radius | Neighborhood size in metres |
| 02 · Demand model | Minimum POIs per zone | DBSCAN core-point threshold |
| 03 · Scoring weights | Five sliders | Relative importance of each criterion; auto-normalized |
| 04 · Shortlist | Candidates to shortlist | 3–8 sites |
| 04 · Shortlist | Minimum site spacing | Spatial suppression distance |
| 04 · Shortlist | Max sites per district | Prevents stacking candidates in one zone |

Changing any control reruns the model immediately.

### Views

- **Opportunity map** — the main decision surface. Numbered pins are shortlisted
  candidates; dashed rings show 900 m straight-line coverage. The right rail previews
  the top candidates and explains the top score.
- **Demand zones** — inspect the DBSCAN result, zone demand ranking, dominant POI types
  and the demand chart.
- **Site shortlist** — full ranked table, criterion comparison, score composition per
  candidate, and a CSV download.
- **Method & data** — formula, current weights, data lineage, assumptions, and an
  optional raw POI/road/charger map.

The 900 m rings are **straight-line buffers**, not drive-time catchments. They are
useful for comparing spatial coverage, not for claiming a specific drive-time market.

---

## Setup and run

### Prerequisites

- Python 3.9 or newer
- Internet access for OpenStreetMap downloads and CARTO basemap tiles
- No API key, account or secret is required

### 1. Create a virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Start the dashboard

```bash
streamlit run app.py
```

Open <http://localhost:8501>. To use another port:

```bash
streamlit run app.py --server.port 8502
```

For a headless server:

```bash
streamlit run app.py --server.headless true --server.address 0.0.0.0
```

### 4. First load

The first OSM load can take several seconds to a minute, depending on Overpass
availability. On success it writes a cache file under `.cache/`; subsequent loads are
fast. If Overpass is unavailable, the dashboard
automatically switches to the deterministic simulated POI/road dataset and shows a
fallback status in the hero banner.

To work without any OSM network calls, choose **Simulated · offline** in the sidebar.
Basemap tiles still require internet; without them the overlays and tables still work,
but the map background may be blank.

---

## Configuration

Most defaults live near the top of each module.

| Setting | Location | Default |
| --- | --- | --- |
| Study-area bounding box | `data_loader.AUSTIN_BBOX` | `(30.18, -97.83, 30.38, -97.66)` |
| Grid cell size | `data_loader.GRID_STEP_DEG` | `0.005°` (~550 m) |
| OSM cache lifetime | `data_loader.OSM_CACHE_TTL_SECONDS` | 24 hours |
| Overpass endpoints | `data_loader.OVERPASS_URLS` | Main API + two mirrors |
| POI demand weights | `data_loader.POI_WEIGHTS` | mall 3.0, marketplace 2.4, commercial 1.9, parking 1.0 |
| DBSCAN defaults | `model.cluster_demand_zones()` | 550 m, 5 POIs |
| Scoring defaults | `model.DEFAULT_WEIGHTS` | 30/20/15/15/20 |
| Accessibility decay | `model.score_grid()` | 700 m |
| Competition radius | `model.score_grid()` | 1,600 m |
| Shortlist defaults | `model.shortlist_sites()` | 5 sites, 1.2 km spacing, 2 per zone |
| Study-area zoom cap | `map_viz._base_map()` | zoom 13 |
| Streamlit data cache | `app.cached_load()` | 12 hours |

To change cities, update `AUSTIN_BBOX`, `AUSTIN_CENTER`, `CITY_NAME` and the
`AUSTIN_NEIGHBORHOODS` seed list in `data_loader.py`. The model is metric-based, so it
will work in any UTM zone after changing `METRIC_CRS` to the appropriate zone.

---

## Data, caching and offline behavior

| Layer | Source | Real or simulated |
| --- | --- | --- |
| Commercial POIs | OpenStreetMap Overpass API | Real OSM data |
| Major-road access points | OpenStreetMap Overpass API | Real OSM data |
| Existing EV chargers | `generate_mock_ev_stations()` | Simulated |
| Population density | `generate_demographic_grid()` | Simulated |
| Income strength | `generate_demographic_grid()` | Simulated |
| Demand zones | DBSCAN over POIs | Derived |
| Opportunity scores | Five-criteria model | Derived |

Caching works in two layers:

1. **Disk cache** — `.cache/osm_<bbox>.json` for 24 hours. The sidebar's
   **Refresh OSM cache** button deletes these files and clears Streamlit's data cache.
2. **Streamlit cache** — `load_all_data()` results are cached for 12 hours.

The simulated fallback is deterministic (fixed random seeds), so repeated offline runs
are stable and comparable.

OpenStreetMap data is © OpenStreetMap contributors. Map tiles are provided by CARTO.
Attribution is included inside each map.

---

## Troubleshooting

**The app says “Simulated fallback” or “OSM unavailable”**

The Overpass API can be rate-limited or unreachable. Wait and press **Refresh OSM
cache**, or continue with the deterministic simulation. The model and UI are unchanged.

**Map tiles are blank**

The CARTO basemap requires internet access. The Folium overlays, tables and scoring
still render. Check network access to `basemaps.cartocdn.com`.

**“Address already in use”**

Another Streamlit process owns port 8501. Run on a different port:

```bash
streamlit run app.py --server.port 8502
```

**A macOS `NotOpenSSLWarning` appears**

This comes from the system Python's LibreSSL/urllib3 combination. It is a warning, not
an app failure. Using a current Python from python.org or Homebrew avoids it.

**Maps or tables look narrow after resizing the browser**

Hard-refresh the page. The app conditionally renders only the selected view; a normal
load should size components to their containers.

**Dependency installation fails**

Confirm you are inside the project virtual environment:

```bash
which python
python --version
```

Then reinstall the pins:

```bash
pip install --force-reinstall -r requirements.txt
```

---

## Verification

Basic syntax check:

```bash
python -m py_compile app.py data_loader.py model.py map_viz.py
```

Offline pipeline smoke test:

```bash
python - <<'PY'
from data_loader import load_all_data
from model import cluster_demand_zones, score_grid, shortlist_sites

data = load_all_data(source="simulated")
clusters = cluster_demand_zones(data["pois"], 550, 5)
scored = score_grid(data["grid"], data["pois"], data["roads"], data["ev_stations"], clusters)
shortlist = shortlist_sites(scored, clusters, 5, 1200, 2)
print(data["meta"])
print(f"{len(clusters)} zones, {len(scored)} cells, {len(shortlist)} shortlisted sites")
PY
```

Expected: the script completes without exceptions and reports a non-empty cluster set,
one score per grid cell and up to five shortlisted sites.

A typical cached run in this workspace produced approximately 2,900 OSM POIs, 5,500
road access points, 1,360 grid cells, 34 DBSCAN zones and a 5-site shortlist. Live
counts will differ as OSM changes.

---

## Limitations and next steps

- Demographics and competitor chargers are synthetic. Replace them with census/ACS
  data and a real charger registry (for example, NREL AFDC) for production use.
- Competition is measured by straight-line distance and station count, not utilization,
  queueing or pricing.
- Coverage rings ignore road barriers, drive times and turn restrictions.
- The model does not include grid capacity, rent, zoning, parking ownership, traffic
  counts or construction cost. Validate any shortlisted parcel before committing capital.
- OSM completeness varies by neighborhood and by data vintage.
- The score is relative to this study area; it is not comparable across cities without
  recalibration.

Natural extensions: drive-time isochrones, parcel-level economics, real demographic and
charger feeds, and automated tests around clustering and shortlist invariants.

---

## Data credits

- POIs and roads: © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright),
  via the [Overpass API](https://overpass-api.de/).
- Basemap tiles: [CARTO](https://carto.com/attributions).
- Clustering and scaling: [scikit-learn](https://scikit-learn.org/).
- Geospatial processing: [GeoPandas](https://geopandas.org/) and
  [Shapely](https://shapely.readthedocs.io/).
- Mapping: [Folium](https://python-visualization.github.io/folium/) and
  [streamlit-folium](https://folium.streamlit.app/).

This is a planning prototype. It is not investment, legal, engineering or utility
advice.
