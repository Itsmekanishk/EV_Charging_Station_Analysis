"""Data acquisition for ChargeMap Intelligence.

Everything the planner needs to reason about Austin, Texas lives here:

* POIs (retail / parking / commercial zones) and major-road access points
  (junctions, signals and road-segment centres) are pulled from OpenStreetMap
  through the Overpass API, with an on-disk cache and a deterministic simulated
  fallback so the app always renders.
* Existing EV charging stations (competitors) and the demographic grid are
  simulated, as specified for the planner prototype.

All frames use WGS84 (EPSG:4326) and expose ``lat`` / ``lon`` helper columns.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point, Polygon, box

try:  # overpy is the primary transport; requests is a hardened fallback
    import overpy
except ImportError:  # pragma: no cover - dependency is pinned in requirements
    overpy = None

# --------------------------------------------------------------------------- #
# Geography & constants
# --------------------------------------------------------------------------- #

#: (south, west, north, east) — central Austin from Zilker to Mueller.
AUSTIN_BBOX: tuple[float, float, float, float] = (30.18, -97.83, 30.38, -97.66)
AUSTIN_CENTER: tuple[float, float] = (30.2747, -97.7431)
CITY_NAME = "Austin, Texas"

GRID_STEP_DEG = 0.005  # ~550 m cells
POPULATION_GRID_STEP = GRID_STEP_DEG

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
#: Fallback mirrors. The official endpoint is preferred; mirrors keep the app
#: usable when the main instance is at capacity or blocked by an older TLS stack.
OVERPASS_URLS = [
    OVERPASS_URL,
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
OVERPASS_HEADERS = {
    "User-Agent": "ChargeMapIntelligence/1.0 (Streamlit EV location-planning prototype)",
    "Accept": "application/json",
}
OSM_CACHE_TTL_SECONDS = 60 * 60 * 24  # 24 h
CACHE_DIR = Path(__file__).resolve().parent / ".cache"

POI_WEIGHTS: dict[str, float] = {
    "mall": 3.0,
    "marketplace": 2.4,
    "commercial_zone": 1.9,
    "parking": 1.0,
}

#: Hand-curated Austin districts used to seed realistic simulated data and to
#: give DBSCAN clusters human-readable names.
AUSTIN_NEIGHBORHOODS: list[dict[str, Any]] = [
    {"name": "Downtown", "lat": 30.2672, "lon": -97.7431, "pois": 11},
    {"name": "East Austin", "lat": 30.2640, "lon": -97.7185, "pois": 9},
    {"name": "South Congress", "lat": 30.2490, "lon": -97.7500, "pois": 7},
    {"name": "Rainey Street", "lat": 30.2570, "lon": -97.7395, "pois": 6},
    {"name": "Zilker", "lat": 30.2590, "lon": -97.7710, "pois": 5},
    {"name": "Westlake", "lat": 30.2795, "lon": -97.7950, "pois": 5},
    {"name": "Tarrytown", "lat": 30.2920, "lon": -97.7780, "pois": 5},
    {"name": "Clarksville", "lat": 30.2760, "lon": -97.7570, "pois": 5},
    {"name": "Hyde Park", "lat": 30.3050, "lon": -97.7330, "pois": 6},
    {"name": "UT Campus", "lat": 30.2849, "lon": -97.7341, "pois": 8},
    {"name": "Mueller", "lat": 30.2980, "lon": -97.7040, "pois": 8},
    {"name": "East Riverside", "lat": 30.2380, "lon": -97.7300, "pois": 6},
    {"name": "North Loop", "lat": 30.3150, "lon": -97.7250, "pois": 5},
    {"name": "Brentwood", "lat": 30.3200, "lon": -97.7400, "pois": 5},
    {"name": "Holly", "lat": 30.2530, "lon": -97.7260, "pois": 5},
    {"name": "Govalle", "lat": 30.2600, "lon": -97.7040, "pois": 4},
    {"name": "Windsor Park", "lat": 30.2940, "lon": -97.7100, "pois": 4},
    {"name": "St. Edwards", "lat": 30.2280, "lon": -97.7550, "pois": 4},
    {"name": "Barton Hills", "lat": 30.2500, "lon": -97.7800, "pois": 3},
    {"name": "Airport Blvd", "lat": 30.2350, "lon": -97.7150, "pois": 5},
]

#: Population centres (lat, lon, amplitude, sigma_m) for the simulated grid.
_POP_CENTERS = [
    (30.2710, -97.7440, 1.00, 1500),  # Downtown
    (30.2849, -97.7341, 0.85, 1200),  # UT
    (30.2600, -97.7180, 0.80, 1700),  # East Austin
    (30.2390, -97.7300, 0.55, 1400),  # Riverside
    (30.2500, -97.7500, 0.50, 1000),  # Soco
    (30.2980, -97.7040, 0.50, 1200),  # Mueller
    (30.3120, -97.7280, 0.45, 1500),  # North Loop
    (30.2780, -97.7950, 0.10, 1800),  # Westlake (low density)
    (30.2280, -97.7550, 0.35, 1400),  # St. Edwards
]

#: Affluent areas for the simulated income field: (lat, lon, amp, sigma_m).
_INCOME_CENTERS = [
    (30.2800, -97.7960, 1.00, 1800),  # Westlake / Tarrytown
    (30.2930, -97.7770, 0.75, 1100),  # Tarrytown
    (30.2680, -97.7590, 0.60, 900),  # Clarksville
    (30.2580, -97.7780, 0.55, 1300),  # Barton Hills
    (30.2960, -97.7400, 0.40, 1600),  # Hyde Park
    (30.2480, -97.7680, 0.35, 1300),  # Zilker
    (30.3300, -97.7450, 0.45, 2200),  # Brentwood / Allandale
    (30.2520, -97.7100, -0.45, 1500),  # East Austin (lower incomes)
    (30.2350, -97.7300, -0.35, 1500),  # Riverside
    (30.2850, -97.7340, -0.25, 900),  # UT students
]
_MOCK_ROAD_PATHS: list[tuple[str, list[tuple[float, float]]]] = [
    ("Interstate 35", [(30.180, -97.737), (30.240, -97.735), (30.300, -97.732), (30.380, -97.727)]),
    ("MoPac Expressway", [(30.180, -97.786), (30.240, -97.779), (30.300, -97.772), (30.380, -97.765)]),
    ("Ben White Blvd (SH-71)", [(30.222, -97.830), (30.225, -97.770), (30.228, -97.710), (30.230, -97.660)]),
    ("US-290", [(30.318, -97.830), (30.322, -97.770), (30.325, -97.720), (30.326, -97.660)]),
    ("Riverside Dr", [(30.240, -97.790), (30.244, -97.745), (30.246, -97.700), (30.248, -97.660)]),
    ("Martin Luther King Blvd", [(30.280, -97.830), (30.282, -97.780), (30.284, -97.730), (30.286, -97.680)]),
    ("Airport Blvd", [(30.180, -97.719), (30.240, -97.716), (30.300, -97.713), (30.360, -97.709)]),
    ("Lamar Blvd", [(30.180, -97.761), (30.240, -97.755), (30.300, -97.748), (30.360, -97.740)]),
    ("South Congress Ave", [(30.180, -97.753), (30.230, -97.750), (30.260, -97.745), (30.280, -97.742)]),
    ("Cesar Chavez St", [(30.248, -97.790), (30.251, -97.750), (30.254, -97.710), (30.256, -97.670)]),
]

_OPERATORS = ["ChargePoint", "Electrify America", "EVgo", "Tesla Supercharger", "Blink"]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _haversine_m(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    """Vectorised great-circle distance in metres from arrays to a point."""
    radius = 6_371_000.0
    phi1, phi2 = np.radians(lat1), math.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * math.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * radius * np.arcsin(np.sqrt(a))


def _field_from_centers(
    lats: np.ndarray, lons: np.ndarray, centers: list[tuple[float, float, float, float]]
) -> np.ndarray:
    """Sum of Gaussian kernels over a lat/lon mesh."""
    values = np.zeros_like(lats, dtype=float)
    for clat, clon, amplitude, sigma in centers:
        dist = _haversine_m(lats, lons, clat, clon)
        values += amplitude * np.exp(-(dist**2) / (2.0 * sigma**2))
    return values


def _normalise_0_100(values: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi - lo < 1e-9:
        return np.full_like(values, 50.0, dtype=float)
    return 100.0 * (values - lo) / (hi - lo)


def _bbox_dict(bbox: tuple[float, float, float, float]) -> dict[str, float]:
    south, west, north, east = bbox
    return {"south": south, "west": west, "north": north, "east": east}


def bbox_as_gdf(bbox: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    south, west, north, east = bbox
    return gpd.GeoDataFrame({"name": ["Austin study area"]}, geometry=[box(west, south, east, north)], crs="EPSG:4326")


# --------------------------------------------------------------------------- #
# Simulated data (also the offline fallback)
# --------------------------------------------------------------------------- #


def generate_mock_pois(
    bbox: tuple[float, float, float, float] = AUSTIN_BBOX,
    seed: int = 42,
) -> gpd.GeoDataFrame:
    """Generate believable commercial POIs around Austin's activity districts."""
    rng = np.random.default_rng(seed)
    south, west, north, east = bbox
    rows: list[dict[str, Any]] = []

    dist_choices = np.array(["parking", "commercial_zone", "mall", "marketplace"])
    dist_probs = np.array([0.44, 0.26, 0.16, 0.14])
    name_suffix = {
        "parking": ["Parking Garage", "Surface Lot", "Park & Ride"],
        "commercial_zone": ["Commercial District", "Retail Corridor", "Business Park"],
        "mall": ["Shopping Center", "Galleria", "Retail Plaza"],
        "marketplace": ["Market", "Public Market", "Farmers Market"],
    }

    def add_point(name: str, category: str, lat: float, lon: float) -> None:
        if south <= lat <= north and west <= lon <= east:
            rows.append(
                {
                    "name": name,
                    "category": category,
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                }
            )

    for district in AUSTIN_NEIGHBORHOODS:
        # Enough points per district for DBSCAN to form a meaningful core, but
        # tight enough that neighbouring districts stay distinct at 550 m.
        n_pois = int(rng.integers(max(5, district["pois"] - 1), district["pois"] + 6))
        for _ in range(n_pois):
            category = str(rng.choice(dist_choices, p=dist_probs))
            lat = district["lat"] + rng.normal(0, 0.0022)
            lon = district["lon"] + rng.normal(0, 0.0026)
            suffix = str(rng.choice(name_suffix[category]))
            add_point(f"{district['name']} {suffix}", category, lat, lon)

    # Background POIs keep the periphery from looking artificially empty.
    for i in range(40):
        category = str(rng.choice(dist_choices, p=dist_probs))
        suffix = str(rng.choice(name_suffix[category]))
        add_point(
            f"North Austin {suffix}" if i % 2 else f"South Austin {suffix}",
            category,
            rng.uniform(south, north),
            rng.uniform(west, east),
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=["name", "category", "lat", "lon"])
    frame.insert(0, "poi_id", [f"SIM-{i:04d}" for i in range(len(frame))])
    frame["source"] = "Simulated"
    return gpd.GeoDataFrame(
        frame,
        geometry=[Point(xy) for xy in zip(frame["lon"], frame["lat"])],
        crs="EPSG:4326",
    )


def generate_mock_roads(
    bbox: tuple[float, float, float, float] = AUSTIN_BBOX,
) -> gpd.GeoDataFrame:
    """Generate major-road access points by densifying known Austin arterials."""
    south, west, north, east = bbox
    rows: list[dict[str, Any]] = []
    spacing_m = 280.0
    for road_name, path in _MOCK_ROAD_PATHS:
        for (lat_a, lon_a), (lat_b, lon_b) in zip(path[:-1], path[1:]):
            length_m = float(_haversine_m(np.array([lat_a]), np.array([lon_a]), lat_b, lon_b)[0])
            steps = max(1, int(length_m // spacing_m))
            for step in range(steps + 1):
                frac = step / steps
                lat = lat_a + frac * (lat_b - lat_a)
                lon = lon_a + frac * (lon_b - lon_a)
                if south <= lat <= north and west <= lon <= east:
                    rows.append({"name": road_name, "category": "major_road", "lat": round(lat, 6), "lon": round(lon, 6)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=["name", "category", "lat", "lon"])
    frame.insert(0, "road_id", [f"RD-{i:04d}" for i in range(len(frame))])
    frame["source"] = "Simulated"
    return gpd.GeoDataFrame(
        frame,
        geometry=[Point(xy) for xy in zip(frame["lon"], frame["lat"])],
        crs="EPSG:4326",
    )


def generate_mock_ev_stations(
    bbox: tuple[float, float, float, float] = AUSTIN_BBOX,
    n_stations: int = 42,
    seed: int = 7,
) -> gpd.GeoDataFrame:
    """Simulate incumbent charging stations, concentrated in busy districts."""
    rng = np.random.default_rng(seed)
    south, west, north, east = bbox
    weights = np.array([district["pois"] for district in AUSTIN_NEIGHBORHOODS], dtype=float)
    weights /= weights.sum()
    rows: list[dict[str, Any]] = []
    for i in range(n_stations):
        if i < int(n_stations * 0.8):
            district = AUSTIN_NEIGHBORHOODS[int(rng.choice(len(AUSTIN_NEIGHBORHOODS), p=weights))]
            base_lat, base_lon = district["lat"], district["lon"]
        else:
            base_lat, base_lon = rng.uniform(south, north), rng.uniform(west, east)
        lat = float(np.clip(base_lat + rng.normal(0, 0.004), south, north))
        lon = float(np.clip(base_lon + rng.normal(0, 0.005), west, east))
        rows.append(
            {
                "station_id": f"EV-{i + 1:03d}",
                "name": f"{str(rng.choice(_OPERATORS))} — {district['name'] if i < int(n_stations * 0.8) else 'Austin'}",
                "operator": str(rng.choice(_OPERATORS)),
                "power_kw": int(rng.choice([7, 11, 22, 50, 150, 350])),
                "stalls": int(rng.integers(2, 17)),
                "lat": round(lat, 6),
                "lon": round(lon, 6),
            }
        )
    frame = pd.DataFrame(rows)
    return gpd.GeoDataFrame(
        frame,
        geometry=[Point(xy) for xy in zip(frame["lon"], frame["lat"])],
        crs="EPSG:4326",
    )


def generate_demographic_grid(
    bbox: tuple[float, float, float, float] = AUSTIN_BBOX,
    step_deg: float = GRID_STEP_DEG,
    seed: int = 11,
) -> gpd.GeoDataFrame:
    """Build a gridded mock demographic surface (population + income scores)."""
    south, west, north, east = bbox
    rng = np.random.default_rng(seed)

    lat_edges = np.arange(south, north, step_deg)
    lon_edges = np.arange(west, east, step_deg)
    if lat_edges[-1] < north - 1e-9:
        lat_edges = np.append(lat_edges, north)
    if lon_edges[-1] < east - 1e-9:
        lon_edges = np.append(lon_edges, east)

    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2.0
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2.0
    grid_lats, grid_lons = np.meshgrid(lat_centers, lon_centers, indexing="ij")
    grid_lats = grid_lats.ravel()
    grid_lons = grid_lons.ravel()

    pop_raw = _field_from_centers(grid_lats, grid_lons, _POP_CENTERS)
    income_raw = _field_from_centers(grid_lats, grid_lons, _INCOME_CENTERS)

    # Correlated smooth noise lets the surface breathe without looking random.
    noise_pop = np.zeros_like(pop_raw)
    noise_income = np.zeros_like(income_raw)
    for _ in range(7):
        clat, clon = rng.uniform(south, north), rng.uniform(west, east)
        sigma = rng.uniform(900, 2600)
        noise_pop += rng.normal(0, 0.10) * np.exp(-(_haversine_m(grid_lats, grid_lons, clat, clon) ** 2) / (2 * sigma**2))
        noise_income += rng.normal(0, 0.09) * np.exp(-(_haversine_m(grid_lats, grid_lons, clat, clon) ** 2) / (2 * sigma**2))
    pop_raw = np.maximum(pop_raw + noise_pop, 0.02)
    income_raw = income_raw + noise_income

    population_score = _normalise_0_100(pop_raw)
    income_score = _normalise_0_100(income_raw)
    population_density = np.round(400 + 14_000 * (population_score / 100.0) ** 1.35)

    cell_ids: list[str] = []
    cells: list[Polygon] = []
    rows: list[dict[str, Any]] = []
    idx = 0
    for i in range(len(lat_centers)):
        for j in range(len(lon_centers)):
            cell_id = f"G-{i:03d}-{j:03d}"
            cell_ids.append(cell_id)
            cells.append(
                box(
                    lon_edges[j],
                    lat_edges[i],
                    lon_edges[j + 1],
                    lat_edges[i + 1],
                )
            )
            rows.append(
                {
                    "cell_id": cell_id,
                    "lat": round(float(lat_centers[i]), 6),
                    "lon": round(float(lon_centers[j]), 6),
                    "population_density": int(population_density[idx]),
                    "population_score": round(float(population_score[idx]), 2),
                    "income_score": round(float(income_score[idx]), 2),
                }
            )
            idx += 1

    return gpd.GeoDataFrame(rows, geometry=cells, crs="EPSG:4326")


# --------------------------------------------------------------------------- #
# OpenStreetMap acquisition
# --------------------------------------------------------------------------- #


def _build_overpass_query(bbox: tuple[float, float, float, float], timeout: int) -> str:
    south, west, north, east = bbox
    b = f"{south:.5f},{west:.5f},{north:.5f},{east:.5f}"
    return f"""
[out:json][timeout:{timeout}];
(
  node["amenity"="parking"]({b});
  way["amenity"="parking"]({b});
  node["shop"="mall"]({b});
  way["shop"="mall"]({b});
  node["amenity"="marketplace"]({b});
  way["amenity"="marketplace"]({b});
  node["landuse"="commercial"]({b});
  way["landuse"="commercial"]({b});
  node["highway"~"^(motorway_junction|traffic_signals)$"]({b});
  way["highway"~"^(motorway|trunk|primary|secondary)$"]({b});
);
out center tags;
""".strip()


def _category_for_tags(tags: dict[str, str]) -> str | None:
    if "highway" in tags:
        return "major_road"
    if tags.get("shop") == "mall":
        return "mall"
    if tags.get("amenity") == "marketplace":
        return "marketplace"
    if tags.get("landuse") == "commercial":
        return "commercial_zone"
    if tags.get("amenity") == "parking":
        return "parking"
    return None


def _rows_from_overpass_elements(elements: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    poi_rows: list[dict[str, Any]] = []
    road_rows: list[dict[str, Any]] = []
    for element in elements:
        tags = element.get("tags") or {}
        category = _category_for_tags(tags)
        if category is None:
            continue
        if element.get("type") == "node":
            lat, lon = element.get("lat"), element.get("lon")
        else:
            center = element.get("center") or {}
            lat, lon = center.get("lat"), center.get("lon")
        if lat is None or lon is None:
            continue
        if category == "major_road":
            road_rows.append(
                {
                    "name": tags.get("name") or tags.get("ref") or "Major road",
                    "category": "major_road",
                    "lat": round(float(lat), 6),
                    "lon": round(float(lon), 6),
                }
            )
        else:
            poi_rows.append(
                {
                    "name": tags.get("name") or tags.get("operator") or tags.get("brand") or category.replace("_", " ").title(),
                    "category": category,
                    "lat": round(float(lat), 6),
                    "lon": round(float(lon), 6),
                }
            )
    return pd.DataFrame(poi_rows), pd.DataFrame(road_rows)


def _fetch_via_overpy(query: str, timeout: int) -> list[dict[str, Any]]:
    if overpy is None:
        raise RuntimeError("overpy is not installed")
    api = overpy.Overpass(url=OVERPASS_URL, max_retry_count=1)
    result = api.query(query)
    elements: list[dict[str, Any]] = []
    for node in result.nodes:
        elements.append({"type": "node", "lat": node.lat, "lon": node.lon, "tags": dict(node.tags)})
    for way in result.ways:
        center_lat = getattr(way, "center_lat", None)
        center_lon = getattr(way, "center_lon", None)
        if center_lat is None and getattr(way, "nodes", None):
            resolved = [n for n in way.nodes if getattr(n, "lat", None) is not None]
            if resolved:
                center_lat = sum(n.lat for n in resolved) / len(resolved)
                center_lon = sum(n.lon for n in resolved) / len(resolved)
        if center_lat is None:
            continue
        elements.append({"type": "way", "lat": center_lat, "lon": center_lon, "tags": dict(way.tags)})
    return elements


def _fetch_via_requests(query: str) -> list[dict[str, Any]]:
    last_error: Exception | None = None
    for url in OVERPASS_URLS:
        try:
            response = requests.post(
                url,
                data={"data": query},
                headers=OVERPASS_HEADERS,
                timeout=(8, 60),
            )
            response.raise_for_status()
            payload = response.json()
            elements = payload.get("elements", [])
            if elements:
                return elements
            last_error = RuntimeError(f"{url} returned no elements")
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            last_error = exc
    raise RuntimeError(f"All Overpass endpoints failed: {last_error}")



def _cache_path(bbox: tuple[float, float, float, float]) -> Path:
    south, west, north, east = bbox
    key = f"{south:.3f}_{west:.3f}_{north:.3f}_{east:.3f}"
    return CACHE_DIR / f"osm_{key}.json"


def _read_cache(bbox: tuple[float, float, float, float]) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    path = _cache_path(bbox)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        if time.time() - payload.get("fetched_unix", 0) > OSM_CACHE_TTL_SECONDS:
            return None
        pois = pd.DataFrame(payload.get("pois", []))
        roads = pd.DataFrame(payload.get("roads", []))
        if pois.empty:
            return None
        return pois, roads
    except (OSError, ValueError, KeyError):
        return None


def _write_cache(bbox: tuple[float, float, float, float], pois: pd.DataFrame, roads: pd.DataFrame) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(bbox).write_text(
            json.dumps(
                {
                    "fetched_unix": time.time(),
                    "pois": pois.to_dict(orient="records"),
                    "roads": roads.to_dict(orient="records"),
                }
            )
        )
    except OSError:
        pass  # caching is best-effort; never fail a load because of it


def fetch_osm_features(
    bbox: tuple[float, float, float, float] = AUSTIN_BBOX,
    timeout: int = 45,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Fetch POIs and major-road access points, returning (pois, roads, status)."""
    if use_cache:
        cached = _read_cache(bbox)
        if cached is not None:
            cached_pois, cached_roads = cached
            cached_pois["source"] = "OpenStreetMap"
            cached_roads["source"] = "OpenStreetMap"
            return cached_pois, cached_roads, "OpenStreetMap (cached)"

    query = _build_overpass_query(bbox, timeout)
    last_error: Exception | None = None
    for fetcher, label in ((_fetch_via_overpy, "overpy"), (_fetch_via_requests, "requests")):
        try:
            elements = fetcher(query, timeout) if label == "overpy" else fetcher(query)  # type: ignore[call-arg]
            pois, roads = _rows_from_overpass_elements(elements)
            if pois.empty:
                last_error = RuntimeError("Overpass returned no POIs")
                continue
            pois.insert(0, "poi_id", [f"OSM-{i:05d}" for i in range(len(pois))])
            roads.insert(0, "road_id", [f"RD-{i:05d}" for i in range(len(roads))])
            pois["source"] = "OpenStreetMap"
            roads["source"] = "OpenStreetMap"
            _write_cache(bbox, pois, roads)
            return pois, roads, "OpenStreetMap (live)"
        except Exception as exc:  # noqa: BLE001 - network failures must degrade gracefully
            last_error = exc
            continue

    return pd.DataFrame(), pd.DataFrame(), f"OSM unavailable ({last_error.__class__.__name__ if last_error else 'unknown error'})"


# --------------------------------------------------------------------------- #
# Public loader
# --------------------------------------------------------------------------- #


def _as_geo(points: pd.DataFrame, id_column: str, id_prefix: str) -> gpd.GeoDataFrame:
    points = points.copy()
    if points.empty:
        points = pd.DataFrame(columns=[id_column, "name", "category", "lat", "lon", "source"])
    if id_column not in points.columns:
        points.insert(0, id_column, [f"{id_prefix}-{i:04d}" for i in range(len(points))])
    points["lat"] = points["lat"].astype(float)
    points["lon"] = points["lon"].astype(float)
    return gpd.GeoDataFrame(
        points,
        geometry=[Point(xy) for xy in zip(points["lon"], points["lat"])],
        crs="EPSG:4326",
    )


def load_all_data(
    bbox: tuple[float, float, float, float] = AUSTIN_BBOX,
    source: str = "auto",
    seed: int = 42,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Load every layer the dashboard needs.

    ``source`` is one of ``auto`` (live OSM if reachable, simulated otherwise),
    ``live`` (same, but explicitly requested) or ``simulated`` (no network).
    """
    bbox = tuple(float(v) for v in bbox)  # type: ignore[assignment]
    south, west, north, east = bbox
    if not (south < north and west < east):
        raise ValueError("Invalid bounding box: expected (south, west, north, east)")

    meta: dict[str, Any] = {
        "city": CITY_NAME,
        "bbox": _bbox_dict(bbox),
        "requested_source": source,
        "loaded_at": _utc_now(),
    }

    pois = roads = None
    if source == "simulated":
        meta["status"] = "Simulated data"
    else:
        pois, roads, status = fetch_osm_features(bbox, use_cache=use_cache)
        meta["osm_status"] = status
        if pois is not None and not pois.empty:
            meta["status"] = "OpenStreetMap" if "live" in status else status
        else:
            pois, roads = None, None

    if pois is None or pois.empty:
        pois = generate_mock_pois(bbox, seed=seed)
        roads = generate_mock_roads(bbox)
        if source == "simulated":
            meta["status"] = "Simulated data"
        elif "osm_status" not in meta:
            meta["status"] = "Simulated fallback"
        elif "unavailable" in str(meta.get("osm_status", "")) or "no POIs" in str(meta.get("osm_status", "")):
            meta["status"] = "Simulated fallback (OSM unavailable)"
        else:
            meta["status"] = "Simulated fallback"
        meta["fallback"] = True
    else:
        meta["fallback"] = False

    pois = _as_geo(pois, "poi_id", "POI")
    roads = _as_geo(roads, "road_id", "RD")
    ev_stations = generate_mock_ev_stations(bbox, seed=max(seed, 7))
    grid = generate_demographic_grid(bbox, step_deg=GRID_STEP_DEG, seed=seed + 101)
    study_area = bbox_as_gdf(bbox)

    meta["counts"] = {
        "pois": int(len(pois)),
        "roads": int(len(roads)),
        "stations": int(len(ev_stations)),
        "grid_cells": int(len(grid)),
    }
    meta["category_counts"] = pois["category"].value_counts().to_dict()

    return {
        "pois": pois,
        "roads": roads,
        "ev_stations": ev_stations,
        "grid": grid,
        "study_area": study_area,
        "meta": meta,
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    bundle = load_all_data(source="simulated")
    print(bundle["meta"])
    for key in ("pois", "roads", "ev_stations", "grid"):
        print(key, len(bundle[key]), bundle[key].crs)
