"""
Choropleth rendering of Allen atlas slice GeoJSONs with Plotly.

Atlas coordinates are pixels, not real longitude/latitude as Coropleth expects. 
We first convert the pixels coords into a "fake" lon/lat coordinates so
go.Choroplethmapbox accepts them. 

@author Anna Teruel-Sanchis, May 2026
"""

from __future__ import annotations
import copy
import os

import plotly.graph_objects as go
from plotlybrain.render import (
    ScoreName,
    get_color_scale_params,
    load_geojson,
    load_score,
)

def _collect_coords(
        coords: list
        ) -> tuple[list[float], list[float]]:
    """
    Recursively collect all (x, y) pairs from a GeoJSON coordinate tree.
    """
    xs, ys = [], []
    for item in coords:
        if isinstance(item[0], (int, float)):
            xs.append(item[0])
            ys.append(item[1])
        else:
            sub_x, sub_y = _collect_coords(item)
            xs.extend(sub_x)
            ys.extend(sub_y)
    return xs, ys


def _rescale_coords(
    coords: list,
    x_min: float,
    x_span: float,
    y_min: float,
    y_span: float,
    lon_lo: float,
    lon_hi: float,
    lat_lo: float,
    lat_hi: float,
) -> list:
    """
    Recursively rescale pixel coordinates to pseudo lon/lat values.

    The Y axis is flipped because pixel Y increases downward while latitude
    increases upward.
    """
    out = []
    for item in coords:
        if isinstance(item[0], (int, float)):
            lon = lon_lo + (item[0] - x_min) / x_span * (lon_hi - lon_lo)
            lat = lat_hi - (item[1] - y_min) / y_span * (lat_hi - lat_lo)
            out.append([lon, lat])
        else:
            out.append(_rescale_coords(
                item,
                x_min, x_span, y_min, y_span,
                lon_lo, lon_hi, lat_lo, lat_hi,
            ))
    return out


def rescale_geojson_to_pseudogeo(
    geojson_obj: dict,
    lon_range: tuple[float, float] = (-1.0, 1.0),
    lat_range: tuple[float, float] = (-1.0, 1.0),
) -> tuple[dict, dict[str, tuple[float, float]]]:
    """
    Linearly rescale pixel coordinates in a GeoJSON FeatureCollection so they
    fall within a small pseudo-geographic lon/lat bounding box.

    go.Choroplethmapbox requires lon/lat coordinates. Allen atlas GeoJSONs
    produced by build_selected_slices() use pixel space. This function remaps
    pixel coordinates to a tight pseudo-geographic box, preserving the relative
    shape of every region. No real geographic meaning is implied.

    Args:
        geojson_obj : dict
            GeoJSON FeatureCollection in pixel space.
        lon_range : tuple[float, float]
            Target longitude span. Default: (-1, 1).
        lat_range : tuple[float, float]
            Target latitude span. Default: (-1, 1).

    Returns:
        tuple[dict, dict]
            - Rescaled GeoJSON FeatureCollection (deep copy).
            - Bounds dict: ``{"lon": (lo, hi), "lat": (lo, hi)}``, used to
              set the mapbox center and zoom.
    """
    all_x, all_y = [], []
    for feat in geojson_obj.get("features", []):
        xs, ys = _collect_coords(feat["geometry"]["coordinates"])
        all_x.extend(xs)
        all_y.extend(ys)

    if not all_x:
        return geojson_obj, {"lon": lon_range, "lat": lat_range}

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_span = x_max - x_min or 1.0
    y_span = y_max - y_min or 1.0

    lon_lo, lon_hi = lon_range
    lat_lo, lat_hi = lat_range

    rescaled = copy.deepcopy(geojson_obj)
    for feat in rescaled["features"]:
        feat["geometry"]["coordinates"] = _rescale_coords(
            feat["geometry"]["coordinates"],
            x_min, x_span, y_min, y_span,
            lon_lo, lon_hi, lat_lo, lat_hi,
        )

    bounds = {"lon": (lon_lo, lon_hi), "lat": (lat_lo, lat_hi)}
    return rescaled, bounds


# ---------------------------------------------------------------------------
# Feature-ID stamping
# ---------------------------------------------------------------------------

def _stamp_feature_ids(geojson_obj: dict, col_id: str = "Region ID") -> dict:
    """
    Set the top-level ``id`` field on every GeoJSON feature to its Region ID.

    go.Choroplethmapbox matches features to data rows via the feature ``id``
    field. Allen atlas GeoJSONs store the region ID inside ``properties``;
    this function promotes it to the top level as a string.

    Args:
        geojson_obj : dict
            GeoJSON FeatureCollection (modified in place).
        col_id : str
            Property key whose value is used as the feature ID.

    Returns:
        dict
            The same FeatureCollection with ``id`` set on every feature.
    """
    for feat in geojson_obj.get("features", []):
        rid = feat.get("properties", {}).get(col_id)
        if rid is not None:
            feat["id"] = str(int(rid))
    return geojson_obj


# ---------------------------------------------------------------------------
# Main rendering function
# ---------------------------------------------------------------------------

def render_brain_slice_choropleth(
    geojson_obj: dict,
    id2value: dict[int, float],
    id2name: dict[int, str] | None = None,
    title: str | None = None,
    score_name: str = "score",
    colorscale: str | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
    col_id: str = "Region ID",
    lon_range: tuple[float, float] = (-1.0, 1.0),
    lat_range: tuple[float, float] = (-1.0, 1.0),
) -> go.Figure:
    """
    Render one Allen atlas slice as a Plotly choropleth using a single
    go.Choroplethmapbox trace.

    Unlike the go.Scatter-based approach (one trace per polygon), this
    function produces a single trace for all regions. Plotly handles color
    scaling and the colorbar automatically.

    Pixel coordinates in the GeoJSON are rescaled to pseudo lon/lat so that
    go.Choroplethmapbox can accept them. A plain white background is used;
    no Mapbox token is required.

    Args:
        geojson_obj : dict
            Slice GeoJSON FeatureCollection as built by build_selected_slices().
        id2value : dict[int, float]
            Region ID → score value.
        id2name : dict[int, str] | None
            Region ID → display name (used in hover labels).
        title : str | None
            Figure title.
        score_name : str
            Score column name, used as the colorbar title and in hover text.
        colorscale : str | None
            Plotly colorscale name. Inferred from score_name if not provided.
        zmin, zmax : float | None
            Color normalization bounds. Inferred from score_name if not provided.
        col_id : str
            Property key in the GeoJSON that holds the region ID.
        lon_range, lat_range : tuple[float, float]
            Pseudo-geographic bounding box for pixel-coordinate rescaling.

    Returns:
        go.Figure
            Plotly figure with a single go.Choroplethmapbox trace.
    """
    if colorscale is None and zmin is None and zmax is None:
        colorscale, zmin, zmax = get_color_scale_params(score_name)
    elif colorscale is None:
        colorscale = "Viridis"

    rescaled_geo, bounds = rescale_geojson_to_pseudogeo(
        geojson_obj, lon_range=lon_range, lat_range=lat_range,
    )
    _stamp_feature_ids(rescaled_geo, col_id=col_id)

    locations: list[str] = []
    z_values: list[float] = []
    hover_texts: list[str] = []

    for feat in rescaled_geo.get("features", []):
        rid_str = feat.get("id")
        if rid_str is None:
            continue
        rid = int(rid_str)
        if rid not in id2value:
            continue

        val = id2value[rid]
        name = (id2name or {}).get(rid, str(rid))
        locations.append(rid_str)
        z_values.append(val)
        hover_texts.append(
            f"Region ID: {rid}<br>Region: {name}<br>{score_name}: {val:.4g}"
        )

    center_lon = (bounds["lon"][0] + bounds["lon"][1]) / 2
    center_lat = (bounds["lat"][0] + bounds["lat"][1]) / 2

    trace_kwargs: dict = dict(
        geojson=rescaled_geo,
        locations=locations,
        z=z_values,
        colorscale=colorscale,
        marker_opacity=1.0,
        marker_line_width=0.5,
        marker_line_color="rgba(255,255,255,0.9)",
        text=hover_texts,
        hoverinfo="text",
        colorbar=dict(title=score_name),
        name="",
    )
    if zmin is not None:
        trace_kwargs["zmin"] = zmin
    if zmax is not None:
        trace_kwargs["zmax"] = zmax

    fig = go.Figure(go.Choroplethmapbox(**trace_kwargs))

    fig.update_layout(
        title=title,
        mapbox=dict(
            style="white-bg",
            center=dict(lon=center_lon, lat=center_lat),
            zoom=5,
            layers=[],
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="white",
    )

    return fig


# ---------------------------------------------------------------------------
# File-based convenience wrapper
# ---------------------------------------------------------------------------

def render_brain_slice_choropleth_from_file(
    score_csv: str,
    geojson_path: str,
    score: ScoreName | None = None,
    value_col: str | None = None,
    title: str | None = None,
    colorscale: str | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
    col_id: str = "Region ID",
    name_col: str = "Region name",
    lon_range: tuple[float, float] = (-1.0, 1.0),
    lat_range: tuple[float, float] = (-1.0, 1.0),
) -> go.Figure:
    """
    Load a score CSV and a slice GeoJSON from disk, then render a choropleth.

    This is the file-based convenience wrapper for
    render_brain_slice_choropleth(), mirroring the interface of
    render_brain_slice_from_file() in render.py.

    Args:
        score_csv : str
            Path to a region-level score CSV.
        geojson_path : str
            Path to a slice GeoJSON produced by build_selected_slices().
        score : {"rel_abundance", "frequency", "density"} | None
            Optional score type used to infer the value column.
        value_col : str | None
            Explicit score column name. Overrides score if provided.
        title : str | None
            Figure title. Defaults to "<csv basename> | <geojson basename>".
        colorscale : str | None
            Plotly colorscale name.
        zmin, zmax : float | None
            Color normalization bounds.
        col_id : str
            Region ID column name in both the CSV and the GeoJSON properties.
        name_col : str
            Region name column name in the CSV.
        lon_range, lat_range : tuple[float, float]
            Pseudo-geographic bounding box for pixel-coordinate rescaling.

    Returns:
        go.Figure
            Plotly choropleth figure.
    """
    id2value, id2name, resolved_col = load_score(
        score_csv=score_csv,
        id_col=col_id,
        name_col=name_col,
        value_col=value_col,
        score=score,
    )
    geojson_obj = load_geojson(geojson_path)

    if title is None:
        title = f"{os.path.basename(score_csv)} | {os.path.basename(geojson_path)}"

    return render_brain_slice_choropleth(
        geojson_obj=geojson_obj,
        id2value=id2value,
        id2name=id2name,
        title=title,
        score_name=resolved_col,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        col_id=col_id,
        lon_range=lon_range,
        lat_range=lat_range,
    )