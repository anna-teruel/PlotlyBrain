"""
Choropleth rendering of Allen atlas slice GeoJSONs with Plotly.
@author Anna Teruel-Sanchis, June 2026
"""

from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from plotlybrain.io import load_geojson, load_score

def render_brain_slice(
    geojson_obj: dict,
    score_df: pd.DataFrame,
    value_col: str,
    title: str | None = None,
    cmap: str = "Viridis",
    zmin: float | None = None,
    zmax: float | None = None,
    col_id: str = "Region ID",
    name_col: str = "Region name",
    line_color: str = "rgba(255,255,255,0.9)",
    line_width: float = 0.5,
    map_style: str = "white-bg",
    zoom: float = 3,
    center: dict[str, float] = {"lat": 0, "lon": 0},
    width: int | None = None,
    height: int | None = None,
    exclude_ids: tuple[int, ...] = (0, 997),
) -> go.Figure:
    """
    Render Allen atlas slices using px.choropleth_map.

    Args:
        geojson_obj : dict
            GeoJSON FeatureCollection containing atlas polygons in pseudo
            lon/lat coordinates. Each feature must contain
            ``properties["feature_id"]``.
        score_df : pd.DataFrame
            Region-level score table. Must contain ``col_id`` and
            ``value_col``.
        value_col : str
            Name of the score column to visualize.
        title : str | None, default=None
            Figure title.
        cmap : str, default="Viridis"
            Plotly colorscale used to color regions.
        zmin : float | None, default=None
            Minimum value used for color normalization. If None, Plotly
            infers the minimum from the data.
        zmax : float | None, default=None
            Maximum value used for color normalization. If None, Plotly
            infers the maximum from the data.
        col_id : str, default="Region ID"
            Column/property containing Allen structure IDs.
        name_col : str, default="Region name"
            Column/property containing region names.
        line_color : str, default="rgba(255,255,255,0.9)"
            Boundary color used to outline regions.
        line_width : float, default=0.5
            Boundary line width in pixels.
        map_style : str, default="white-bg"
            Plotly map style.
        zoom : float, default=3
            Initial map zoom level.
        center :dict[str, float] = {"lat": 0, "lon": 0},
            Map center. If None, defaults to ``{"lat": 0, "lon": 0}``.
        width : int | None, default=None
            Figure width in pixels.
        height : int | None, default=None
            Figure height in pixels.
        exclude_ids : tuple[int, ...], default=(0, 997)
            Allen structure IDs excluded from rendering.

    Returns:
        go.Figure
            Plotly choropleth figure.
    """
    score_df = score_df.copy()
    score_df[col_id] = pd.to_numeric(score_df[col_id], errors="coerce").astype("Int64")
    score_df[value_col] = pd.to_numeric(score_df[value_col], errors="coerce")
    score_df = score_df.dropna(subset=[col_id])
    score_df = score_df[~score_df[col_id].isin(exclude_ids)]

    feature_rows = []

    for feat in geojson_obj.get("features", []):
        props = feat.get("properties", {})
        rid = props.get(col_id)

        if rid is None:
            continue

        rid = int(rid)

        if rid in exclude_ids:
            continue

        if "feature_id" not in props:
            raise KeyError(
                "GeoJSON feature is missing properties['feature_id']. "
                "Add it in build_geojson(), for example: "
                "feature_id = f'{slice_index}_{rid}'."
            )

        feature_rows.append(
            {
                "feature_id": props["feature_id"],
                col_id: rid,
                name_col: props.get(name_col, str(rid)),
                "slice_index": props.get("slice_index"),
                "coordinate_mm": props.get("coordinate_mm"),
                "orientation": props.get("orientation"),
                "resolution_um": props.get("resolution_um"),
            }
        )

    feature_df = pd.DataFrame(feature_rows)

    plot_df = feature_df.merge(
        score_df[[col_id, value_col]],
        on=col_id,
        how="left",
    )

    range_color = None
    if zmin is not None and zmax is not None:
        range_color = (zmin, zmax)

    fig = px.choropleth_map(
        plot_df,
        geojson=geojson_obj,
        locations="feature_id",
        color=value_col,
        featureidkey="properties.feature_id",
        hover_name=name_col,
        hover_data={
            col_id: True,
            "slice_index": True,
            "coordinate_mm": True,
            "orientation": True,
            "resolution_um": True,
            value_col: True,
            "feature_id": False,
        },
        color_continuous_scale=cmap,
        range_color=range_color,
        map_style=map_style,
        center=center,
        zoom=zoom,
        width=width,
        height=height,
    )

    fig.update_traces(
        marker_opacity=1.0,
        marker_line_color=line_color,
        marker_line_width=line_width,
    )

    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig