"""Folium map builders for ChargeMap Intelligence."""

from __future__ import annotations

import html
from typing import Any, Iterable

import branca.colormap as cm
import folium
import geopandas as gpd
import pandas as pd

from data_loader import AUSTIN_BBOX

SCORE_COLORS = ["#1e2a4a", "#1d4ed8", "#0891b2", "#10b981", "#facc15"]
SCORE_COLORMAP = cm.LinearColormap(
    colors=SCORE_COLORS,
    vmin=0,
    vmax=100,
    caption="Opportunity score",
)

TIER_OPACITY = {"Prime": 0.62, "Strong": 0.42, "Moderate": 0.24, "Low": 0.10}
TIER_COLORS = {"Prime": "#facc15", "Strong": "#22d3ee", "Moderate": "#818cf8", "Low": "#475569"}

CATEGORY_COLORS = {
    "mall": "#f59e0b",
    "marketplace": "#f472b6",
    "commercial_zone": "#38bdf8",
    "parking": "#94a3b8",
    "major_road": "#64748b",
}
CATEGORY_LABELS = {
    "mall": "Mall / retail",
    "marketplace": "Marketplace",
    "commercial_zone": "Commercial zone",
    "parking": "Parking",
    "major_road": "Major road",
}

# Public tiles, no API key or account required. The dark styling stays in the
# application shell while the map remains usable in offline/demo deployments
# through the cached feature layers.
TILES = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_ATTRIBUTION = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    south, west, north, east = bbox
    return (south + north) / 2.0, (west + east) / 2.0


def _base_map(bbox: tuple[float, float, float, float], zoom: int = 13) -> folium.Map:
    fmap = folium.Map(
        location=_bbox_center(bbox),
        zoom_start=zoom,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
        zoom_control=True,
    )
    folium.TileLayer(
        tiles=TILES,
        attr=TILE_ATTRIBUTION,
        name="CARTO dark basemap",
        subdomains="abcd",
        max_zoom=20,
        detect_retina=True,
        control=True,
    ).add_to(fmap)
    south, west, north, east = bbox
    # max_zoom matters: Streamlit builds hidden tabs with a zero-width container,
    # so Leaflet's fitBounds would otherwise clamp to max zoom and open the
    # Demand-zone map on a few city blocks. Cap it at neighbourhood scale.
    fmap.fit_bounds([[south, west], [north, east]], padding=(18, 18), max_zoom=13)
    return fmap


def _legend_html(title: str, items: Iterable[tuple[str, str]], note: str | None = None) -> str:
    rows = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;">'
        f'<span style="width:11px;height:11px;border-radius:3px;background:{color};'
        f'box-shadow:0 0 0 1px rgba(255,255,255,.18);flex:0 0 auto;"></span>'
        f'<span style="font-size:11.5px;color:#dbe7f6;">{html.escape(label)}</span>'
        f"</div>"
        for label, color in items
    )
    note_html = (
        f'<div style="margin-top:7px;font-size:10px;line-height:1.35;color:#94a3b8;">{html.escape(note)}</div>'
        if note
        else ""
    )
    return f"""
    <div style="position:fixed;bottom:22px;left:22px;z-index:9999;background:rgba(8,15,28,.92);
                border:1px solid rgba(148,163,184,.24);border-radius:12px;padding:12px 14px;
                font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
                backdrop-filter:blur(8px);min-width:158px;max-width:230px;box-shadow:0 8px 24px rgba(0,0,0,.35);">
      <div style="font-size:11px;letter-spacing:.10em;text-transform:uppercase;color:#7dd3fc;
                  font-weight:700;margin-bottom:7px;">{html.escape(title)}</div>
      {rows}{note_html}
    </div>
    """


def _add_legend(fmap: folium.Map, legend: str) -> None:
    fmap.get_root().html.add_child(folium.Element(legend))


def _add_brand_badge(fmap: folium.Map, label: str) -> None:
    badge = f"""
    <div style="position:fixed;top:18px;right:18px;z-index:9999;background:rgba(8,15,28,.88);
                border:1px solid rgba(148,163,184,.24);border-radius:999px;padding:6px 12px;
                font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
                font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:#9fb3cc;">
      {html.escape(label)}
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(badge))


def _score_color(score: float) -> str:
    return SCORE_COLORMAP(max(0.0, min(100.0, float(score))))


def _competitor_markers(fmap: folium.Map, ev_stations: gpd.GeoDataFrame | None, group: str = "Existing chargers") -> None:
    if ev_stations is None or ev_stations.empty:
        return
    layer = folium.FeatureGroup(name=group, show=True)
    for _, row in ev_stations.iterrows():
        operator = html.escape(str(row.get("operator", "Operator")))
        power = row.get("power_kw", "—")
        stalls = row.get("stalls", "—")
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4.5,
            color="#fb7185",
            weight=1.4,
            fill=True,
            fill_color="#fb7185",
            fill_opacity=0.85,
            tooltip=folium.Tooltip(
                f'<div style="font-family:-apple-system,sans-serif;font-size:12px;">'
                f"<b>{operator}</b><br>{power} kW · {stalls} stalls</div>",
                sticky=False,
            ),
        ).add_to(layer)
    layer.add_to(fmap)


def _candidate_marker(target: Any, row: pd.Series, color: str, coverage_radius_m: float) -> None:
    rank = int(row.get("rank", 1))
    score = float(row.get("opportunity_score", 0.0))
    zone = html.escape(str(row.get("zone", "Austin")))
    pin_html = f"""
    <div style="display:flex;flex-direction:column;align-items:center;transform:translate(-50%,-50%);">
      <div style="width:38px;height:38px;border-radius:50%;
                  background:{color};border:2px solid rgba(255,255,255,.92);
                  display:flex;align-items:center;justify-content:center;
                  color:#08111f;font-weight:800;font-size:15px;
                  box-shadow:0 3px 12px rgba(0,0,0,.55);font-family:-apple-system,sans-serif;">{rank}</div>
      <div style="margin-top:2px;padding:2px 6px;border-radius:999px;background:rgba(8,15,28,.9);
                  color:#e2f3ff;font-size:10px;font-weight:700;white-space:nowrap;
                  font-family:-apple-system,sans-serif;border:1px solid rgba(255,255,255,.14);">
        {score:.1f}
      </div>
    </div>
    """
    if coverage_radius_m > 0:
        folium.Circle(
            location=[row["lat"], row["lon"]],
            radius=coverage_radius_m,
            color=color,
            weight=1.4,
            opacity=0.7,
            fill=True,
            fill_color=color,
            fill_opacity=0.055,
            dash_array="5 7",
        ).add_to(target)
    folium.Marker(
        location=[row["lat"], row["lon"]],
        icon=folium.DivIcon(
            html=pin_html,
            icon_size=(80, 80),
            icon_anchor=(40, 40),
            class_name="cm-candidate-pin",
        ),
        tooltip=folium.Tooltip(
            f'<div style="font-family:-apple-system,sans-serif;font-size:12px;line-height:1.5;">'
            f"<b>#{rank} · {zone}</b><br>Opportunity score: <b>{score:.1f}</b><br>"
            f"Nearest charger: {int(row.get('nearest_ev_m', 0))} m · "
            f"{int(row.get('ev_within_radius', 0))} nearby</div>",
            sticky=True,
        ),
        zindex_offset=100,
    ).add_to(target)


# --------------------------------------------------------------------------- #
# Public maps
# --------------------------------------------------------------------------- #


def create_opportunity_map(
    scored_grid: gpd.GeoDataFrame,
    shortlist: gpd.GeoDataFrame | None = None,
    ev_stations: gpd.GeoDataFrame | None = None,
    clusters: gpd.GeoDataFrame | None = None,
    bbox: tuple[float, float, float, float] = AUSTIN_BBOX,
    coverage_radius_m: float = 900.0,
) -> folium.Map:
    """Main decision map: opportunity heat surface + shortlisted candidate pins."""
    fmap = _base_map(bbox, zoom=12)
    _add_brand_badge(fmap, "ChargeMap · opportunity surface")

    if scored_grid is not None and not scored_grid.empty:
        tooltip_fields = [
            "cell_id",
            "opportunity_score",
            "tier",
            "demand_score",
            "nearest_ev_m",
            "ev_within_radius",
        ]
        tooltip_aliases = [
            "Grid cell:",
            "Opportunity:",
            "Tier:",
            "Demand:",
            "Nearest charger (m):",
            "Chargers nearby:",
        ]

        def style(feature: dict[str, Any]) -> dict[str, Any]:
            props = feature.get("properties", {})
            score = float(props.get("opportunity_score") or 0.0)
            tier = str(props.get("tier") or "Low")
            return {
                "fillColor": _score_color(score),
                "fillOpacity": TIER_OPACITY.get(tier, 0.1),
                "color": TIER_COLORS.get(tier, "#334155"),
                "weight": 1.0 if tier == "Prime" else 0.35,
                "opacity": 0.75 if tier == "Prime" else 0.15,
            }

        def highlight(feature: dict[str, Any]) -> dict[str, Any]:
            return {"fillOpacity": 0.78, "weight": 1.4, "color": "#e0f2fe"}

        folium.GeoJson(
            scored_grid[["cell_id", "opportunity_score", "tier", "demand_score", "nearest_ev_m", "ev_within_radius", "geometry"]],
            name="Opportunity grid",
            style_function=style,
            highlight_function=highlight,
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=tooltip_aliases,
                localize=True,
                sticky=False,
                labels=True,
                style=(
                    "background:rgba(8,15,28,.94);color:#e2e8f0;border:1px solid #334155;"
                    "border-radius:8px;font-family:-apple-system,sans-serif;font-size:12px;padding:8px 10px;"
                ),
            ),
            smooth_factor=0.4,
        ).add_to(fmap)

    if clusters is not None and not clusters.empty:
        cluster_layer = folium.FeatureGroup(name="DBSCAN zones", show=False)
        for _, row in clusters.iterrows():
            color = _score_color(float(row.get("demand_index", 50)))
            folium.CircleMarker(
                location=[row["centroid_lat"], row["centroid_lon"]],
                radius=6,
                color=color,
                weight=1.5,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                tooltip=f'{html.escape(str(row["name"]))} · {int(row["poi_count"])} POIs',
            ).add_to(cluster_layer)
        cluster_layer.add_to(fmap)

    _competitor_markers(fmap, ev_stations)

    if shortlist is not None and not shortlist.empty:
        candidates = folium.FeatureGroup(name="Shortlisted sites + coverage", show=True)
        for _, row in shortlist.iterrows():
            color = _score_color(float(row["opportunity_score"]))
            _candidate_marker(candidates, row, color, coverage_radius_m)
        candidates.add_to(fmap)

    _add_legend(
        fmap,
        _legend_html(
            "Opportunity score",
            [("Prime (top 10%)", "#facc15"), ("Strong", "#22d3ee"), ("Moderate", "#818cf8"), ("Low", "#475569")],
            note="Numbered pins = shortlist rank. Dashed rings = 900 m coverage. Pink dots = existing chargers (simulated).",
        ),
    )
    folium.LayerControl(collapsed=True, position="bottomright").add_to(fmap)
    return fmap


def create_cluster_map(
    pois: gpd.GeoDataFrame,
    clusters: gpd.GeoDataFrame,
    ev_stations: gpd.GeoDataFrame | None = None,
    bbox: tuple[float, float, float, float] = AUSTIN_BBOX,
) -> folium.Map:
    """DBSCAN zones with their member POIs, sized and coloured by demand."""
    fmap = _base_map(bbox, zoom=12)
    _add_brand_badge(fmap, f"ChargeMap · {len(clusters)} demand zones")

    if clusters is not None and not clusters.empty:
        zone_layer = folium.FeatureGroup(name="Demand zones (DBSCAN)", show=True)
        for _, row in clusters.iterrows():
            color = _score_color(float(row.get("demand_index", 50)))
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda _f, c=color: {
                    "fillColor": c,
                    "fillOpacity": 0.17,
                    "color": c,
                    "weight": 1.4,
                    "opacity": 0.8,
                },
                tooltip=folium.Tooltip(
                    f'<div style="font-family:-apple-system,sans-serif;font-size:12px;line-height:1.5;">'
                    f'<b>{html.escape(str(row["name"]))}</b><br>'
                    f'{int(row["poi_count"])} POIs · demand index {float(row["demand_index"]):.0f}/100<br>'
                    f'Dominant: {html.escape(str(row["dominant_category"]).replace("_", " "))}</div>',
                    sticky=True,
                ),
            ).add_to(zone_layer)

            label_html = f"""
            <div style="transform:translate(-50%,-50%);text-align:center;white-space:nowrap;
                        font-family:-apple-system,sans-serif;pointer-events:none;">
              <div style="width:28px;height:28px;margin:0 auto;border-radius:50%;background:rgba(8,15,28,.94);
                          border:2px solid {color};color:#f8fafc;font-size:10px;font-weight:800;
                          display:flex;align-items:center;justify-content:center;">{int(row["poi_count"])}</div>
              <div style="margin-top:3px;display:inline-block;padding:2px 7px;border-radius:5px;
                          background:rgba(8,15,28,.86);color:#dbeafe;font-size:10px;font-weight:600;
                          border:1px solid rgba(148,163,184,.25);">{html.escape(str(row["name"]))}</div>
            </div>
            """
            folium.Marker(
                location=[row["centroid_lat"], row["centroid_lon"]],
                icon=folium.DivIcon(html=label_html, icon_size=(160, 60), icon_anchor=(80, 30)),
                zindex_offset=20,
            ).add_to(zone_layer)
        zone_layer.add_to(fmap)

    if pois is not None and not pois.empty:
        poi_layer = folium.FeatureGroup(name="Commercial POIs", show=True)
        for _, row in pois.iterrows():
            color = CATEGORY_COLORS.get(str(row.get("category")), "#94a3b8")
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=3.2,
                color=color,
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=0.75,
                tooltip=folium.Tooltip(
                    f'<div style="font-family:-apple-system,sans-serif;font-size:11px;">'
                    f'{html.escape(str(row.get("name", "POI")))}<br>'
                    f'<span style="color:#94a3b8;">{CATEGORY_LABELS.get(str(row.get("category")), "POI")}</span></div>',
                    sticky=False,
                ),
            ).add_to(poi_layer)
        poi_layer.add_to(fmap)

    _competitor_markers(fmap, ev_stations)
    _add_legend(
        fmap,
        _legend_html(
            "DBSCAN clusters",
            [
                ("Mall / retail", CATEGORY_COLORS["mall"]),
                ("Marketplace", CATEGORY_COLORS["marketplace"]),
                ("Commercial zone", CATEGORY_COLORS["commercial_zone"]),
                ("Parking", CATEGORY_COLORS["parking"]),
                ("Existing charger", "#fb7185"),
            ],
            note="Zone colour and centre badge follow the demand index. Points outside any zone are DBSCAN noise.",
        ),
    )
    folium.LayerControl(collapsed=True, position="bottomright").add_to(fmap)
    return fmap


def create_source_map(
    pois: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    ev_stations: gpd.GeoDataFrame | None = None,
    bbox: tuple[float, float, float, float] = AUSTIN_BBOX,
) -> folium.Map:
    """Inspect the raw acquisition layers before any modelling."""
    fmap = _base_map(bbox, zoom=12)
    _add_brand_badge(fmap, "ChargeMap · raw data layers")

    if pois is not None and not pois.empty:
        poi_layer = folium.FeatureGroup(name="Commercial POIs", show=True)
        for _, row in pois.iterrows():
            color = CATEGORY_COLORS.get(str(row.get("category")), "#94a3b8")
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=3.4,
                color=color,
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                tooltip=html.escape(str(row.get("name", "POI"))),
            ).add_to(poi_layer)
        poi_layer.add_to(fmap)

    if roads is not None and not roads.empty:
        road_layer = folium.FeatureGroup(name="Major-road access points", show=False)
        for _, row in roads.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=1.6,
                color="#64748b",
                weight=0,
                fill=True,
                fill_color="#94a3b8",
                fill_opacity=0.5,
                tooltip=html.escape(str(row.get("name", "Major road"))),
            ).add_to(road_layer)
        road_layer.add_to(fmap)

    _competitor_markers(fmap, ev_stations)
    _add_legend(
        fmap,
        _legend_html(
            "Source layers",
            [
                ("Mall / retail", CATEGORY_COLORS["mall"]),
                ("Marketplace", CATEGORY_COLORS["marketplace"]),
                ("Commercial zone", CATEGORY_COLORS["commercial_zone"]),
                ("Parking", CATEGORY_COLORS["parking"]),
                ("Existing charger", "#fb7185"),
            ],
            note="POIs and road nodes come from OpenStreetMap when available; chargers are simulated competitors.",
        ),
    )
    folium.LayerControl(collapsed=True, position="bottomright").add_to(fmap)
    return fmap
