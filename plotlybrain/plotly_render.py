"""
Render Allen atlas slice GeoJSONs with Plotly using region-level score tables.
@author Anna Teruel-Sanchis, Mar 2026
"""

import json
import math
import os
from typing import Literal

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from shapely.geometry import MultiPolygon, shape

ScoreName = Literal["rel_abundance", "frequency", "density"]

def infer_score_column(
    score_df: pd.DataFrame,
    score: ScoreName | None = None,
) -> str:
    """
    Infer the numeric score column from a saved score table.
    In other words, it infers which of the columns should be 
    used as the score.
    Our scores table can vary, so this is a flexible way to 
    manually set the score name we are expecting to read. 

    Args:
        score_df : pd.DataFrame
            Table loaded from one of the score CSVs.
        score : {"rel_abundance", "frequency", "density"} | None
            Optional score name. If provided, the expected column is checked.

    Returns:
        str
            Name of the numeric score column to visualize.
    """
    expected = {
        "rel_abundance": "relative_abundance_z",
        "frequency": "frequency",
        "density": "density",
    }

    if score is not None:
        col = expected[score]
        if col not in score_df.columns:
            raise KeyError(
                f"Expected score column '{col}' not found. "
                f"Available columns: {list(score_df.columns)}"
            )
        return col

    candidates = [
        "relative_abundance_z",
        "frequency",
        "density",
    ]
    found = [c for c in candidates if c in score_df.columns]

    if len(found) == 1:
        return found[0]

    if len(found) == 0:
        raise ValueError(
            "Could not infer score column. None of these were found: "
            f"{candidates}"
        )

    raise ValueError(
        "Multiple possible score columns found. Please provide `score=` "
        f"explicitly. Found: {found}"
    )

def load_score(
    score_csv: str,
    id_col: str = "Region ID",
    name_col: str = "Region name",
    value_col: str | None = None,
    score: ScoreName | None = None,
) -> tuple[dict[int, float], dict[int, str], str]:
    """
    Load a region-level score CSV and build lookup dictionaries.

    Args:
        score_csv : str
            Path to a CSV file produced by scores.py.
        id_col : str
            Column name containing Allen structure IDs.
        name_col : str
            Column name containing region names.
        value_col : str | None
            Column name containing the score to color by.
            If None, it is inferred from `score` or the table contents.
        score : {"rel_abundance", "frequency", "density"} | None
            Optional score type used to infer the score column.

    Returns:
        tuple[dict[int, float], dict[int, str], str]
            - id2value: structure_id -> score value
            - id2name: structure_id -> region name
            - value_col: resolved score column name
    """
    df = pd.read_csv(score_csv)
    if id_col not in df.columns:
        raise KeyError(
            f"Column '{id_col}' not found in score table. "
            f"Available columns: {list(df.columns)}"
        )

    if name_col not in df.columns:
        raise KeyError(
            f"Column '{name_col}' not found in score table. "
            f"Available columns: {list(df.columns)}"
        )

    if value_col is None:
        value_col = infer_score_column(df, score=score)

    missing = [c for c in [id_col, name_col, value_col] if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df = (
        df[[id_col, name_col, value_col]]
        .assign(
            **{
                id_col: lambda d: pd.to_numeric(d[id_col], errors="coerce").astype("Int64"),
                value_col: lambda d: pd.to_numeric(d[value_col], errors="coerce"),
            }
        )
        .dropna(subset=[id_col])
    )

    id2value = ( #coloring regions dictionary
        df.dropna(subset=[value_col])
        .set_index(id_col)[value_col]
        .astype(float)
        .to_dict()
    )

    id2name = ( #hover labels dictionary
        df.dropna(subset=[name_col])
        .set_index(id_col)[name_col]
        .astype(str)
        .to_dict()
    )

    return id2value, id2name, value_col

def load_geojson(
    geojson_path: str,
) -> dict:
    """
    Load a slice GeoJSON file.

    Args:
        geojson_path : str
            Path to a slice GeoJSON file.

    Returns:
        dict
            GeoJSON FeatureCollection.
    """
    with open(geojson_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_color_scale_params(
    score_name: str,
    cmap: str | None = None, 
) -> tuple[str, float | None, float | None]:
    """
    Choose a default colorscale and limits depending on the score type.

    Args:
        score_name: str
            Score column name.
        cmap: str

    Returns:
        tuple[str, float | None, float | None]
            colorscale, zmin, zmax
    """
    cmap = cmap or "RdBu_r"   

    if score_name == "relative_abundance_z":
        return cmap, -3.0, 3.0

    if score_name == "frequency":
        return cmap, 0.0, 1.0

    if score_name == "density":
        return cmap, None, None

    return "Viridis", None, None

def value_to_color(
    value: float | None,
    vmin: float | None,
    vmax: float | None,
    colorscale: str = "RdBu_r",
    na_color: str = "#d9d9d9",
) -> str:
    """
    Map a numeric value (score) to a Plotly colorscale color.

    Args:
        value : float | None
            Score value for one region.
        vmin, vmax : float | None
            Color normalization limits.
        colorscale : str
            Plotly colorscale name.
        na_color : str
            Fallback color for missing values.

    Returns:
        str
            CSS color string.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return na_color

    if vmin is None or vmax is None or vmax == vmin:
        t = 0.5
    else:
        t = (float(value) - float(vmin)) / (float(vmax) - float(vmin))
        t = max(0.0, min(1.0, t))

    return sample_colorscale(colorscale, [t])[0]

def _add_colorbar_trace(
    fig: go.Figure,
    score_name: str,
    colorscale: str,
    zmin: float | None,
    zmax: float | None,
) -> None:
    """
    Plotly does not automatically generate a colorbar when polygons are
    rendered using individual ``Scatter`` traces with manually assigned
    fill colors. This function adds an invisible marker trace whose sole
    purpose is to expose a colorbar corresponding to the chosen colorscale
    and normalization range.

    Args:
        fig : go.Figure
            Plotly figure to which the colorbar trace will be added.
        score_name : str
            Label displayed as the colorbar title.
        colorscale : str
            Plotly colorscale name used for mapping values to colors.
        zmin : float | None
            Lower bound of the color normalization range. If None, the
            colorscale is displayed without fixed limits.
        zmax : float | None
            Upper bound of the color normalization range. If None, the
            colorscale is displayed without fixed limits.

    Returns:
        None
            The function modifies the input figure in place.
    """
    if zmin is None or zmax is None:
        marker = dict(
            size=0,
            color=[0],
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title=score_name),
        )
    else:
        marker = dict(
            size=0,
            color=[zmin, zmax],
            colorscale=colorscale,
            cmin=zmin,
            cmax=zmax,
            showscale=True,
            colorbar=dict(title=score_name),
        )

    fig.add_trace(
        go.Scatter(
            x=[None] * len(marker["color"]),
            y=[None] * len(marker["color"]),
            mode="markers",
            marker=marker,
            hoverinfo="skip",
            showlegend=False,
        )
    )

def render_brain_slice(
    geojson_obj: dict,
    id2value: dict[int, float],
    id2name: dict[int, str] | None = None,
    title: str | None = None,
    score_name: str = "score",
    colorscale: str | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
    col_id: str = "Region ID",
    name_col: str = "Region name",
    line_color: str = "rgba(255,255,255,0.95)",
    line_width: float = 0.7,
    line_shape: str = "spline",
    smoothing: float = 0.6,
    na_color: str = "#d9d9d9",
    show_colorbar: bool = True,
) -> go.Figure:
    """
    Render one atlas slice as planar Cartesian polygons.

    Args:
        geojson_obj : dict
            Slice GeoJSON object.
        id2value : dict[int, float]
            Region ID -> score value.
        id2name : dict[int, str] | None
            Region ID -> region name.
        title : str | None
            Figure title.
        score_name : str
            Name of the score being plotted.
        colorscale : str | None
            Plotly colorscale.
        zmin, zmax : float | None
            Optional color scale limits.
        col_id : str
            Region ID property name in GeoJSON.
        name_col : str
            Region name property name.
        line_color : str
            Boundary color.
        line_width : float
            Boundary width.
        line_shape : str
            Plotly scatter line shape. Use "spline" for smoother contours.
        smoothing : float
            Spline smoothing factor when line_shape="spline".
        na_color : str
            Fill color for missing values.
        show_colorbar : bool
            Whether to display a colorbar.

    Returns:
        plotly.graph_objects.Figure
            Plotly figure.
    """
    if colorscale is None and zmin is None and zmax is None:
        colorscale, zmin, zmax = get_color_scale_params(score_name)
    elif colorscale is None:
        colorscale = "Viridis"

    fig = go.Figure()

    for feat in geojson_obj.get("features", []):
        props = feat.get("properties", {})
        rid = props.get(col_id)
        if rid is None:
            continue

        rid = int(rid)
        value = id2value.get(rid, None)
        region_name = props.get(name_col, str(rid))
        if id2name is not None:
            region_name = id2name.get(rid, region_name)

        geom = shape(feat["geometry"])
        polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
        fillcolor = value_to_color(value, zmin, zmax, colorscale=colorscale, na_color=na_color)

        for poly in polys:
            x, y = poly.exterior.xy
            fig.add_trace(
                go.Scatter(
                    x=list(x),
                    y=list(y),
                    mode="lines",
                    fill="toself",
                    fillcolor=fillcolor,
                    line=dict(
                        color=line_color,
                        width=line_width,
                        shape=line_shape,
                        smoothing=smoothing,
                    ),
                    customdata=[[rid, region_name, value]] * len(x),
                    hovertemplate=(
                        "Region ID: %{customdata[0]}<br>"
                        "Region: %{customdata[1]}<br>"
                        f"{score_name}: %{{customdata[2]}}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

    if show_colorbar:
        _add_colorbar_trace(
            fig=fig,
            score_name=score_name,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
        )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", autorange="reversed")

    fig.update_layout(
        title=title,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=40, b=10),
    )

    return fig

def render_brain_slice_from_file(
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
    line_color: str = "rgba(255,255,255,0.95)",
    line_width: float = 0.7,
    line_shape: str = "spline",
    smoothing: float = 0.6,
    na_color: str = "#d9d9d9",
    show_colorbar: bool = True,
) -> go.Figure:
    """
    Load a region-level score table and a slice GeoJSON from disk, then render
    a color-coded atlas slice using Plotly using ``render_brain_slice()``function.
    
    Args:
        score_csv : str
            Path to a CSV file containing region-level scores (e.g. frequency,
            density, or relative abundance).
        geojson_path : str
            Path to a GeoJSON file representing a single atlas slice.
        score : {"rel_abundance", "frequency", "density"} | None
            Optional score type used to infer the score column.
        value_col : str | None
            Explicit name of the score column. Overrides ``score`` if provided.
        title : str | None
            Optional figure title. If None, a default title is generated from
            the input file names.
        colorscale : str | None
            Plotly colorscale name used to map values to colors.
        zmin, zmax : float | None
            Optional lower and upper bounds for color normalization.
        col_id : str
            Column name corresponding to region IDs in both the CSV and GeoJSON.
        name_col : str
            Column name corresponding to region names.
        line_color : str
            Boundary color for region outlines.
        line_width : float
            Boundary line width.
        line_shape : str
            Plotly line shape ("linear" or "spline"). "spline" produces smoother
            boundaries.
        smoothing : float
            Spline smoothing factor when ``line_shape="spline"``.
        na_color : str
            Fill color used for regions with missing values.
        show_colorbar : bool
            Whether to display a colorbar.

    Returns:
        plotly.graph_objects.Figure
            Plotly figure showing the atlas slice with regions colored by score.
    """
    id2value, id2name, resolved_value_col = load_score(
        score_csv=score_csv,
        id_col=col_id,
        name_col=name_col,
        value_col=value_col,
        score=score,
    )

    geojson_obj = load_geojson(geojson_path)

    if title is None:
        title = f"{os.path.basename(score_csv)} | {os.path.basename(geojson_path)}"

    return render_brain_slice(
        geojson_obj=geojson_obj,
        id2value=id2value,
        id2name=id2name,
        title=title,
        score_name=resolved_value_col,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        col_id=col_id,
        name_col=name_col,
        line_color=line_color,
        line_width=line_width,
        line_shape=line_shape,
        smoothing=smoothing,
        na_color=na_color,
        show_colorbar=show_colorbar,
    )

def load_manifest(
    manifest_csv: str,
) -> pd.DataFrame:
    """
    Load the slice manifest generated by build_selected_slices().

    Args:
        manifest_csv : str
            Path to slice_manifest.csv.

    Returns:
        pd.DataFrame
            Manifest table.
    """
    df = pd.read_csv(manifest_csv)

    if "slice_index" in df.columns:
        df["slice_index"] = pd.to_numeric(df["slice_index"], errors="coerce").astype("Int64")

    if "ap_mm" in df.columns:
        df["ap_mm"] = pd.to_numeric(df["ap_mm"], errors="coerce")

    return df

def find_geojson_for_slice(
    manifest_df: pd.DataFrame,
    slice_index: int | None = None,
    ap_mm: float | None = None,
) -> str:
    """
    Resolve a GeoJSON path from the manifest using either exact slice index
    or nearest AP coordinate.

    Args:
        manifest_df : pd.DataFrame
            Slice manifest table.
        slice_index : int | None
            Exact slice index to use.
        ap_mm : float | None
            AP coordinate; nearest slice will be selected.

    Returns:
        str
            GeoJSON file path.
    """
    if slice_index is not None:
        sub = manifest_df[manifest_df["slice_index"] == int(slice_index)]
        if sub.empty:
            raise ValueError(f"No slice found for slice_index={slice_index}")
        return str(sub.iloc[0]["geojson_path"])

    if ap_mm is not None:
        if "ap_mm" not in manifest_df.columns:
            raise KeyError("Manifest does not contain 'ap_mm'.")

        valid = manifest_df.dropna(subset=["ap_mm"]).copy()
        if valid.empty:
            raise ValueError("Manifest has no valid ap_mm values.")

        idx = (valid["ap_mm"] - ap_mm).abs().idxmin()
        return str(valid.loc[idx, "geojson_path"])

    raise ValueError("Provide either `slice_index` or `ap_mm`.")

def export_brain_slice(
    fig: go.Figure,
    out_path: str,
    fmt: str | None = None,
    width: int | None = None,
    height: int | None = None,
    scale: float = 1.0,
) -> None:
    """
    Export a Plotly brain slice figure to a static file (SVG, PNG, PDF).

    Args:
        fig : go.Figure
            Plotly figure generated by render_brain_slice().
        out_path : str
            Output file path.
        fmt : {"svg", "png", "pdf"} | None
            Output format. If None, inferred from file extension.
            Defaults to "svg" if no extension is provided.
        width : int | None
            Optional width in pixels.
        height : int | None
            Optional height in pixels.
        scale : float
            Scaling factor for resolution (default = 1.0).

    Returns:
        None
    """
    if fmt is None:
        _, ext = os.path.splitext(out_path)
        if ext:
            fmt = ext.lower().replace(".", "")
        else:
            fmt = "svg"
            out_path = out_path + ".svg"
    else:
        fmt = fmt.lower()
        if not out_path.lower().endswith(f".{fmt}"):
            out_path = f"{out_path}.{fmt}"

    valid_formats = {"svg", "png", "pdf"}
    if fmt not in valid_formats:
        raise ValueError(
            f"Unsupported format '{fmt}'. Supported formats: {valid_formats}"
        )

    try:
        fig.write_image(
            out_path,
            format=fmt,
            width=width,
            height=height,
            scale=scale,
        )
    except ValueError as e:
        raise RuntimeError(
            "Static export requires the 'kaleido' package.\n"
            "Install it with: pip install kaleido"
        ) from e