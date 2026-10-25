"""Spatial modelling for ChargeMap Intelligence.

Two stages:

1. **Demand zones** — DBSCAN groups commercial POIs into dense demand clusters
   (each with a human-readable Austin district label and a demand index).
2. **Multi-criteria site scoring** — every cell of the demographic grid is
   scored on five weighted criteria using MinMaxScaler-normalised values:
   demand potential, population density, household income, road accessibility
   and competition gap.

The application then picks a spatially separated shortlist of top candidates.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from data_loader import AUSTIN_CENTER, AUSTIN_NEIGHBORHOODS, POI_WEIGHTS

#: EPSG:32614 — UTM zone 14N, metres, appropriate for Austin distance maths.
METRIC_CRS = "EPSG:32614"

DEFAULT_WEIGHTS: dict[str, float] = {
    "demand": 0.30,
    "population": 0.20,
    "income": 0.15,
    "accessibility": 0.15,
    "competition": 0.20,
}

CRITERIA_LABELS: dict[str, str] = {
    "demand": "Demand potential",
    "population": "Population density",
    "income": "Income strength",
    "accessibility": "Road accessibility",
    "competition": "Competition gap",
}

CRITERIA_DESCRIPTIONS: dict[str, str] = {
    "demand": "Kernel-smoothed POI demand from DBSCAN zones plus local POI density.",
    "population": "Simulated residential density score for the surrounding grid.",
    "income": "Simulated purchasing-power score — a proxy for EV adoption headroom.",
    "accessibility": "Exponential decay of distance to the nearest major-road access point.",
    "competition": "Distance from existing chargers (65%) plus scarcity of nearby stations (35%).",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _metric(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if frame.crs is None:
        frame = frame.set_crs("EPSG:4326")
    return frame.to_crs(METRIC_CRS)


def _scale_0_100(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
        return np.full(values.shape, 100.0)
    return 100.0 * (values - lo) / (hi - lo)


def _nearest_distance_m(source_xy: np.ndarray, target_xy: np.ndarray, fallback: float = 2500.0) -> np.ndarray:
    """Distance in metres from each source point to its nearest target point."""
    source_xy = np.asarray(source_xy, dtype=float)
    if len(target_xy) == 0 or len(source_xy) == 0:
        return np.full(len(source_xy), fallback, dtype=float)
    tree = NearestNeighbors(n_neighbors=1).fit(np.asarray(target_xy, dtype=float))
    distances, _ = tree.kneighbors(source_xy)
    return distances[:, 0]


def normalise_weights(weights: dict[str, float] | None) -> dict[str, float]:
    """Return weights that sum to one, falling back to defaults if unusable."""
    merged = dict(DEFAULT_WEIGHTS)
    if weights:
        for key in DEFAULT_WEIGHTS:
            if key in weights and weights[key] is not None:
                merged[key] = max(0.0, float(weights[key]))
    total = sum(merged.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {key: value / total for key, value in merged.items()}


# --------------------------------------------------------------------------- #
# DBSCAN demand zones
# --------------------------------------------------------------------------- #


def cluster_demand_zones(
    pois: gpd.GeoDataFrame,
    eps_m: float = 550.0,
    min_samples: int = 5,
) -> gpd.GeoDataFrame:
    """Group commercial POIs into dense demand zones with DBSCAN.

    Returns one row per cluster with a demand index (0–100), radius, dominant
    category, a human-readable Austin district name and a hull geometry.
    """
    columns = [
        "cluster_id",
        "label",
        "name",
        "poi_count",
        "dominant_category",
        "demand_index",
        "centroid_lat",
        "centroid_lon",
        "radius_m",
        "geometry",
    ]
    empty = gpd.GeoDataFrame(columns=columns, geometry="geometry", crs="EPSG:4326")
    empty.attrs.update({"n_clusters": 0, "n_noise": 0, "eps_m": eps_m, "min_samples": min_samples})
    if pois is None or pois.empty or len(pois) < max(2, int(min_samples)):
        return empty

    pois_metric = _metric(pois).reset_index(drop=True)
    coords = np.column_stack([pois_metric.geometry.x, pois_metric.geometry.y])
    labels = DBSCAN(eps=float(eps_m), min_samples=int(min_samples), metric="euclidean").fit_predict(coords)
    pois_metric = pois_metric.assign(_cluster=labels)

    cluster_labels = sorted(int(label) for label in set(labels) if label != -1)
    if not cluster_labels:
        empty.attrs["n_noise"] = int(len(pois_metric))
        return empty

    # Reference points for naming clusters after Austin districts.
    district_frame = gpd.GeoDataFrame(
        {"name": [d["name"] for d in AUSTIN_NEIGHBORHOODS]},
        geometry=[Point(d["lon"], d["lat"]) for d in AUSTIN_NEIGHBORHOODS],
        crs="EPSG:4326",
    ).to_crs(METRIC_CRS)

    city_center = gpd.GeoSeries(
        [Point(AUSTIN_CENTER[1], AUSTIN_CENTER[0])], crs="EPSG:4326"
    ).to_crs(METRIC_CRS).iloc[0]

    rows: list[dict[str, Any]] = []
    geometries: list[Any] = []
    used_names: dict[str, int] = {}
    assigned_names: set[str] = set()
    for label in cluster_labels:
        members = pois_metric[pois_metric["_cluster"] == label]
        points = np.column_stack([members.geometry.x, members.geometry.y])
        centroid = points.mean(axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        radius = float(max(np.percentile(distances, 90) + 120.0, 250.0))

        category_counts = members["category"].value_counts()
        dominant = str(category_counts.index[0]) if len(category_counts) else "parking"
        demand_weight = float(
            sum(POI_WEIGHTS.get(str(cat), 1.0) * int(count) for cat, count in category_counts.items())
        )

        nearest_district, district_distance = _nearest_named_point(
            centroid, district_frame.geometry.x.to_numpy(), district_frame.geometry.y.to_numpy()
        )
        if district_distance > 4500:
            base_name = f"{_compass_direction(centroid[0] - city_center.x, centroid[1] - city_center.y)} Austin corridor"
        else:
            base_name = str(district_frame.iloc[nearest_district]["name"])
        used_names[base_name] = used_names.get(base_name, 0) + 1
        if used_names[base_name] == 1:
            name = base_name
        elif district_distance > 4500:
            name = f"{base_name} ({used_names[base_name]})"
        else:
            anchor = district_frame.iloc[nearest_district].geometry
            direction = _compass_direction(centroid[0] - anchor.x, centroid[1] - anchor.y)
            name = f"{base_name} · {direction}"
            if name in assigned_names:
                name = f"{base_name} · {direction} {used_names[base_name]}"
        assigned_names.add(name)

        cluster_points = gpd.GeoSeries(
            [Point(x, y) for x, y in points], crs=METRIC_CRS
        )
        hull = cluster_points.union_all().convex_hull
        if hull.geom_type != "Polygon" or hull.area < 1:
            hull = Point(centroid).buffer(radius)
        hull = hull.buffer(radius * 0.25).simplify(35)

        centroid_wgs = gpd.GeoSeries([Point(*centroid)], crs=METRIC_CRS).to_crs("EPSG:4326").iloc[0]
        hull_wgs = gpd.GeoSeries([hull], crs=METRIC_CRS).to_crs("EPSG:4326").iloc[0]

        rows.append(
            {
                "cluster_id": f"C{len(rows) + 1}",
                "label": label,
                "name": name,
                "poi_count": int(len(members)),
                "dominant_category": dominant,
                "demand_weight": demand_weight,
                "centroid_lat": round(float(centroid_wgs.y), 6),
                "centroid_lon": round(float(centroid_wgs.x), 6),
                "radius_m": int(round(radius)),
            }
        )
        geometries.append(hull_wgs)

    clusters = gpd.GeoDataFrame(rows, geometry=geometries, crs="EPSG:4326")
    # Log compression prevents one very large OSM cluster (dense downtown parking)
    # from flattening every other district's demand index to zero.
    clusters["demand_index"] = np.round(
        _scale_0_100(np.log1p(clusters["demand_weight"].to_numpy())), 1
    )
    clusters = clusters.sort_values(["demand_index", "poi_count"], ascending=False).reset_index(drop=True)
    clusters["cluster_id"] = [f"C{i + 1}" for i in range(len(clusters))]
    clusters = clusters[columns]
    clusters.attrs.update(
        {
            "n_clusters": int(len(clusters)),
            "n_noise": int((labels == -1).sum()),
            "eps_m": float(eps_m),
            "min_samples": int(min_samples),
        }
    )
    return clusters


def _compass_direction(dx: float, dy: float) -> str:
    """Eight-point compass label for an east/north offset in metres."""
    angle = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
    labels = ["East", "Northeast", "North", "Northwest", "West", "Southwest", "South", "Southeast"]
    return labels[int(((angle + 22.5) % 360.0) // 45.0)]


def _nearest_named_point(point: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> tuple[int, float]:
    if len(xs) == 0:
        return -1, float("inf")
    distances = np.hypot(xs - point[0], ys - point[1])
    idx = int(np.argmin(distances))
    return idx, float(distances[idx])


# --------------------------------------------------------------------------- #
# Multi-criteria scoring
# --------------------------------------------------------------------------- #


def _count_within_radius(
    grid_point: gpd.GeoSeries, targets_metric: gpd.GeoDataFrame, radius_m: float
) -> np.ndarray:
    """Count target points inside a circular neighbourhood of every grid cell."""
    if targets_metric is None or targets_metric.empty:
        return np.zeros(len(grid_point), dtype=int)
    buffers = gpd.GeoDataFrame(
        {"_grid_index": np.arange(len(grid_point))},
        geometry=grid_point.buffer(radius_m).to_numpy(),
        crs=METRIC_CRS,
    )
    joined = gpd.sjoin(
        targets_metric[["geometry"]],
        buffers,
        how="inner",
        predicate="within",
    )
    counts = joined.groupby("_grid_index").size() if len(joined) else pd.Series(dtype=int)
    return counts.reindex(range(len(grid_point)), fill_value=0).to_numpy(dtype=int)


def score_grid(
    grid: gpd.GeoDataFrame,
    pois: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    ev_stations: gpd.GeoDataFrame,
    clusters: gpd.GeoDataFrame,
    weights: dict[str, float] | None = None,
    poi_radius_m: float = 850.0,
    competition_radius_m: float = 1600.0,
    cluster_sigma_m: float = 1200.0,
) -> gpd.GeoDataFrame:
    """Score every grid cell with weighted MinMax-normalised criteria (0–100)."""
    if grid is None or grid.empty:
        return gpd.GeoDataFrame(columns=["cell_id", "opportunity_score", "tier"], geometry=[], crs="EPSG:4326")

    weights_used = normalise_weights(weights)
    scored = grid.copy().reset_index(drop=True)
    grid_metric = _metric(scored)
    points_metric = grid_metric.geometry.centroid
    xy = np.column_stack([points_metric.x, points_metric.y])
    pois_metric = _metric(pois) if pois is not None and not pois.empty else None
    roads_metric = _metric(roads) if roads is not None and not roads.empty else None
    ev_metric = _metric(ev_stations) if ev_stations is not None and not ev_stations.empty else None

    # --- Demand: DBSCAN cluster kernels + local POI density ------------------
    cluster_influence = np.zeros(len(scored), dtype=float)
    if clusters is not None and not clusters.empty:
        cluster_points = gpd.GeoSeries(
            [Point(lon, lat) for lon, lat in zip(clusters["centroid_lon"], clusters["centroid_lat"])],
            crs="EPSG:4326",
        ).to_crs(METRIC_CRS)
        cluster_xy = np.column_stack([cluster_points.x, cluster_points.y])
        demand_weights = clusters["demand_index"].to_numpy(dtype=float) / 100.0
        for (cx, cy), cluster_weight in zip(cluster_xy, demand_weights):
            dist_sq = (xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2
            cluster_influence += cluster_weight * np.exp(-dist_sq / (2.0 * cluster_sigma_m**2))
        cluster_influence = _scale_0_100(cluster_influence)

    poi_count = _count_within_radius(points_metric, pois_metric, poi_radius_m) if pois_metric is not None else np.zeros(len(scored))
    poi_density = _scale_0_100(poi_count.astype(float))
    demand_raw = 0.65 * cluster_influence + 0.35 * poi_density

    # --- Accessibility ------------------------------------------------------
    road_xy = (
        np.column_stack([roads_metric.geometry.x, roads_metric.geometry.y])
        if roads_metric is not None
        else np.empty((0, 2))
    )
    road_dist = _nearest_distance_m(xy, road_xy, fallback=4000.0)
    accessibility_raw = 100.0 * np.exp(-road_dist / 700.0)

    # --- Competition gap ----------------------------------------------------
    ev_xy = (
        np.column_stack([ev_metric.geometry.x, ev_metric.geometry.y])
        if ev_metric is not None
        else np.empty((0, 2))
    )
    ev_dist = _nearest_distance_m(xy, ev_xy, fallback=5000.0)
    ev_count = _count_within_radius(points_metric, ev_metric, competition_radius_m) if ev_metric is not None else np.zeros(len(scored))
    competition_raw = 100.0 * (
        0.65 * _scale_0_100(ev_dist) / 100.0 + 0.35 * (1.0 - _scale_0_100(ev_count.astype(float)) / 100.0)
    )

    population_raw = scored["population_score"].to_numpy(dtype=float)
    income_raw = scored["income_score"].to_numpy(dtype=float)

    raw_matrix = np.column_stack([demand_raw, population_raw, income_raw, accessibility_raw, competition_raw])
    raw_matrix = np.nan_to_num(raw_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    normalised = MinMaxScaler().fit_transform(raw_matrix) * 100.0

    weight_vector = np.array([weights_used[key] for key in DEFAULT_WEIGHTS], dtype=float)
    opportunity = normalised @ weight_vector

    scored["demand_score"] = np.round(normalised[:, 0], 1)
    scored["population_score"] = np.round(normalised[:, 1], 1)
    scored["income_score"] = np.round(normalised[:, 2], 1)
    scored["accessibility_score"] = np.round(normalised[:, 3], 1)
    scored["competition_gap_score"] = np.round(normalised[:, 4], 1)
    scored["opportunity_score"] = np.round(opportunity, 1)
    scored["nearest_road_m"] = np.round(road_dist).astype(int)
    scored["nearest_ev_m"] = np.round(ev_dist).astype(int)
    scored["ev_within_radius"] = ev_count.astype(int)
    scored["poi_within_radius"] = poi_count.astype(int)
    scored["cluster_demand_influence"] = np.round(cluster_influence, 1)

    percentiles = scored["opportunity_score"].rank(pct=True, method="average")
    scored["tier"] = np.select(
        [percentiles >= 0.90, percentiles >= 0.70, percentiles >= 0.40],
        ["Prime", "Strong", "Moderate"],
        default="Low",
    )
    scored["opportunity_rank"] = (
        scored["opportunity_score"].rank(ascending=False, method="first").astype(int)
    )

    scored.attrs.update(
        {
            "weights": weights_used,
            "poi_radius_m": poi_radius_m,
            "competition_radius_m": competition_radius_m,
            "cluster_sigma_m": cluster_sigma_m,
            "score_min": float(scored["opportunity_score"].min()),
            "score_max": float(scored["opportunity_score"].max()),
            "score_mean": float(scored["opportunity_score"].mean()),
        }
    )
    return scored


# Backwards-friendly alias with a clearer verb.
evaluate_locations = score_grid


def shortlist_sites(
    scored_grid: gpd.GeoDataFrame,
    clusters: gpd.GeoDataFrame | None = None,
    n_sites: int = 5,
    min_spacing_m: float = 1200.0,
    max_per_zone: int = 2,
) -> gpd.GeoDataFrame:
    """Pick the best spatially separated cells, enriched with their demand zone.

    Selection is greedy: candidates must respect the minimum spacing *and* the
    per-district cap so the shortlist represents distinct trade areas rather than
    five adjacent cells in the single hottest cluster.
    """
    keep_columns = [
        "cell_id",
        "lat",
        "lon",
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
        "cluster_demand_influence",
        "geometry",
    ]
    if scored_grid is None or scored_grid.empty:
        return gpd.GeoDataFrame(columns=keep_columns + ["rank", "site_name"], geometry="geometry", crs="EPSG:4326")

    ranked = scored_grid.sort_values(
        ["opportunity_score", "population_score"], ascending=False
    ).reset_index(drop=True)
    coords = np.column_stack(
        [
            ranked.to_crs(METRIC_CRS).geometry.centroid.x,
            ranked.to_crs(METRIC_CRS).geometry.centroid.y,
        ]
    )

    # Assign every candidate a district up front so the zone cap can steer selection.
    zone_by_row = np.array(["Austin"] * len(ranked), dtype=object)
    zone_distance_by_row = np.full(len(ranked), np.nan, dtype=float)
    has_zones = clusters is not None and not clusters.empty
    if has_zones:
        cluster_points = gpd.GeoSeries(
            [Point(lon, lat) for lon, lat in zip(clusters["centroid_lon"], clusters["centroid_lat"])],
            crs="EPSG:4326",
        ).to_crs(METRIC_CRS)
        cluster_xy = np.column_stack([cluster_points.x, cluster_points.y])
        dist, pos = NearestNeighbors(n_neighbors=1).fit(cluster_xy).kneighbors(coords)
        for i, (distance, position) in enumerate(zip(dist[:, 0], pos[:, 0])):
            zone_distance_by_row[i] = float(distance)
            zone_by_row[i] = str(clusters.iloc[position]["name"]) if distance <= 4000 else "Underserved corridor"

    zone_cap = max(1, int(max_per_zone)) if has_zones else max(1, int(n_sites))
    selected: list[int] = []
    zone_counts: dict[str, int] = {}
    for idx in range(len(ranked)):
        if len(selected) >= int(n_sites):
            break
        zone = str(zone_by_row[idx])
        # "Underserved corridor" is a loose outer label, not a single district.
        if zone != "Underserved corridor" and zone_counts.get(zone, 0) >= zone_cap:
            continue
        if all(np.hypot(*(coords[idx] - coords[chosen])) >= min_spacing_m for chosen in selected):
            selected.append(idx)
            zone_counts[zone] = zone_counts.get(zone, 0) + 1

    shortlist = ranked.iloc[selected].copy().reset_index(drop=True)
    if shortlist.empty:
        return shortlist

    zone_names = [str(zone_by_row[idx]) for idx in selected]
    zone_distances = [float(zone_distance_by_row[idx]) for idx in selected]

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    shortlist["rank"] = np.arange(1, len(shortlist) + 1)
    shortlist["zone"] = zone_names
    shortlist["distance_to_zone_m"] = [int(round(value)) if np.isfinite(value) else None for value in zone_distances]
    shortlist["site_name"] = [
        f"Site {letters[i]} — {zone}"
        for i, zone in enumerate(zone_names)
    ]
    columns = keep_columns + ["rank", "zone", "distance_to_zone_m", "site_name"]
    shortlist = gpd.GeoDataFrame(shortlist[columns], geometry="geometry", crs="EPSG:4326")
    shortlist.attrs.update(
        {"requested": int(n_sites), "min_spacing_m": min_spacing_m, "max_per_zone": zone_cap}
    )
    return shortlist


def criterion_breakdown(shortlist: gpd.GeoDataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Long-form weighted contribution table for the shortlist charts."""
    weights_used = normalise_weights(weights)
    if shortlist is None or shortlist.empty:
        return pd.DataFrame(columns=["site_name", "criterion", "label", "score", "weight", "weighted_points"])
    column_for = {
        "demand": "demand_score",
        "population": "population_score",
        "income": "income_score",
        "accessibility": "accessibility_score",
        "competition": "competition_gap_score",
    }
    records: list[dict[str, Any]] = []
    for _, row in shortlist.iterrows():
        for key, column in column_for.items():
            records.append(
                {
                    "site_name": row["site_name"],
                    "criterion": key,
                    "label": CRITERIA_LABELS[key],
                    "score": float(row[column]),
                    "weight": weights_used[key],
                    "weighted_points": round(float(row[column]) * weights_used[key], 1),
                }
            )
    return pd.DataFrame(records)
