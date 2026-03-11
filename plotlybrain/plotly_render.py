"""
Interactive Plotly rendering for Allen Brain Atlas SVG section files.
Visualization. 
@author Anna Teruel-Sanchis, Jan 2026
"""

from __future__ import annotations
import os
from typing import Dict, Optional, Union, List
import plotly.graph_objects as go
from .allen_api import download_section_svg, DEFAULT_GROUP_ID

def render_slice_plotly(
    svg_path: str,
    id2value: Dict[int, float],
    id2name: Optional[Dict[int, str]] = None,
    *,
    group_id: int = DEFAULT_GROUP_ID,
    n_samples: int = 120,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorscale: Union[str, list] = "Viridis",
    missing_color: str = "rgba(0,0,0,0)",  # transparent
    show_outlines: bool = True,
    outline_width: float = 0.5,
) -> go.Figure:
    """
    Render a single Allen Brain Atlas SVG section as an interactive Plotly figure.

    The input SVG is expected to be an Allen "structure boundaries" SVG (typically
    graphic_group_28). Each region is represented by one or more <path> elements
    with a structure identifier attribute (structure_id / structureId).

    This function:
      1) parses the SVG and extracts region paths (structure_id + path geometry),
      2) approximates each path as a polygon by sampling points along the SVG curve,
      3) draws each region as a filled Plotly polygon trace (hoverable),
      4) colors each region using id2value[structure_id] and a Plotly colorscale.

    Args:
        svg_path (str):
            Path to the Allen SVG file for one section (slice).
        id2value (Dict[int, float]):
            Mapping from Allen structure_id -> numeric value (e.g., frequency in [0, 1]).
            Regions missing from this mapping are filled with `missing_color`.
        id2name (Dict[int, str], optional):
            Optional mapping structure_id -> region name for hover text.
        group_id (int):
            Allen graphic group ID to use. Default is 28 (structure boundaries).
        n_samples (int):
            Number of sample points taken along each SVG path. Increase for smoother
            boundaries (slower), decrease for speed (rougher).
        vmin (float, optional):
            Minimum value used for colorscale normalization. If None, inferred from
            id2value (ignoring NaNs).
        vmax (float, optional):
            Maximum value used for colorscale normalization. If None, inferred from
            id2value (ignoring NaNs).
        colorscale (str or list):
            Plotly colorscale name (e.g., "Temps", "Viridis") or an explicit Plotly
            colorscale list.
        missing_color (str):
            Fill color used when a region has no value (default transparent).
        show_outlines (bool):
            Whether to draw region outlines.
        outline_width (float):
            Outline width in pixels (only used if show_outlines=True).

    Returns:
        go.Figure:
            Interactive Plotly figure showing the slice. Each region is hoverable and
            filled according to the provided values.
    """
    import math
    import numpy as np
    import plotly.colors as pc
    import plotly.graph_objects as go
    from lxml import etree
    from svgpathtools import parse_path

    SVG_NS = {"svg": "http://www.w3.org/2000/svg"}

    # --- parse SVG ---
    root = etree.parse(svg_path).getroot()

    # --- viewBox: needed to flip y so slice is not upside-down ---
    vb = root.get("viewBox")
    if vb:
        xmin, ymin, w, h = map(float, vb.replace(",", " ").split())
    else:
        # fallback: try width/height (often include px); if missing, use unit square
        def _to_float(v, default):
            if not v:
                return default
            v = v.strip().lower().replace("px", "")
            try:
                return float(v)
            except ValueError:
                return default
        xmin, ymin = 0.0, 0.0
        w = _to_float(root.get("width"), 1.0)
        h = _to_float(root.get("height"), 1.0)

    # --- determine normalization range if not given ---
    vals = []
    for v in id2value.values():
        if v is None:
            continue
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            continue
        vals.append(float(v))
    if vmin is None:
        vmin = min(vals) if vals else 0.0
    if vmax is None:
        vmax = max(vals) if vals else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-9

    def _clamp01(t):
        return 0.0 if t < 0 else (1.0 if t > 1 else t)

    def value_to_color(val):
        if val is None:
            return missing_color
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return missing_color
        t = _clamp01((float(val) - vmin) / (vmax - vmin))
        return pc.sample_colorscale(colorscale, [t])[0]

    # --- restrict to Allen structure boundary group ---
    group = root.find(f".//svg:g[@id='graphic_group_{group_id}']", namespaces=SVG_NS)
    search_root = group if group is not None else root

    fig = go.Figure()

    # --- iterate paths, sample geometry, draw polygons ---
    for p in search_root.findall(".//svg:path", namespaces=SVG_NS):
        sid = p.get("structure_id") or p.get("structureId")
        d = p.get("d")
        if not sid or not d or not sid.isdigit():
            continue
        sid_int = int(sid)

        # sample points along the SVG path
        path = parse_path(d)
        ts = np.linspace(0.0, 1.0, n_samples)
        pts = [path.point(t) for t in ts]
        x = np.array([z.real for z in pts], dtype=float)
        y = np.array([z.imag for z in pts], dtype=float)

        # flip y: SVG y increases downward; Plotly y increases upward
        y = (ymin + h) - y

        val = id2value.get(sid_int, None)
        name = (id2name.get(sid_int) if id2name else None) or f"structure {sid_int}"

        hover = f"{name}<br>ID: {sid_int}"
        hover += f"<br>value: {val:.4g}" if val is not None else "<br>value: (missing)"

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                fill="toself",
                fillcolor=value_to_color(val),
                line=dict(width=outline_width if show_outlines else 0),
                hoverinfo="text",
                text=hover,
                showlegend=False,
                name=name,
            )
        )

    # --- tidy layout: keep aspect ratio and hide axes ---
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False, constrain="domain"),
        yaxis=dict(visible=False, scaleanchor="x"),
    )

    return fig

def render_slice_from_allen(
    section_image_id: int,
    id2value: Dict[int, float],
    id2name: Optional[Dict[int, str]] = None,
    *,
    svg_dir: str = "allen_svgs",
    group_id: int = DEFAULT_GROUP_ID,
    cache: bool = True,
    overwrite: bool = False,
    timeout: int = 60,
    **render_kwargs,
) -> go.Figure:
    """
    Convenience function: download (if needed) and render one Allen atlas slice in Plotly.

    This links `allen_api.py` (download) with `svg_plotly.py` (render), while keeping
    responsibilities separated.

    Args:
        section_image_id (int):
            Allen section image (sub_image) ID for the slice to render.
        id2value (dict[int, float]):
            structure_id -> value map (e.g., frequency).
        id2name (dict[int, str], optional):
            structure_id -> region name (for hover text).
        svg_dir (str):
            Directory used to cache downloaded SVGs.
        group_id (int):
            Allen graphic group ID (default: 28).
        cache (bool):
            If True, reuse existing SVG file if present.
        overwrite (bool):
            If True, force re-download.
        timeout (int):
            Max seconds to wait for the download.
        **render_kwargs:
            Forwarded to `render_slice_plotly()` (e.g., colorscale, vmin, vmax, n_samples).

    Returns:
        plotly.graph_objects.Figure:
            Interactive Plotly figure for this slice.
    """
    os.makedirs(svg_dir, exist_ok=True)
    svg_path = os.path.join(svg_dir, f"{section_image_id}.svg")

    download_section_svg(
        section_image_id=section_image_id,
        out_path=svg_path,
        group_id=group_id,
        cache=cache,
        overwrite=overwrite,
        timeout=timeout,
    )

    return render_slice_plotly(
        svg_path=svg_path,
        id2value=id2value,
        id2name=id2name,
        group_id=group_id,
        **render_kwargs,
    )