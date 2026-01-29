"""
Recolor Allen Brain Atlas sections SVGs using region-level scores
@author Anna Teruel-Sanchis, Jan 2026
"""

from __future__ import annotations

import os
import math
from typing import Dict, Optional, Tuple, List
import pandas as pd
import plotly.express as px
from plotly.colors import sample_colorscale
import xml.etree.ElementTree as ET

from .allen_api import (
    DEFAULT_GROUP_ID,
    download_section_svg,
)

def load_score(
    score_csv: str,
    *,
    id_col: str = "Region ID",
    name_col: str = "Region name",
    value_col: str = "frequency",   
) -> Tuple[Dict[int, float], Dict[int, str]]:
    """
    Load a region-level score CSV and build lookup dictionaries.

    Args:
        score_csv(str): Path to a CSV file produced by scores.py.
        id_col(str): Column name containing Allen structure IDs.
        name_col(str): Column name containing region names.
        value_col(str): Column name containing the score to color by
                        (e.g. 'frequency' or 'relative_abundance_z').

    Returns:
        Tuple[dict[int, float], dict[int, str]]:
            - id2value: structure_id -> score value
            - id2name: structure_id -> region name
    """
    df = (
        pd.read_csv(score_csv)[[id_col, name_col, value_col]]
        .assign(
            **{
                id_col: lambda d: pd.to_numeric(d[id_col], errors="coerce").astype("Int64"),
                value_col: lambda d: pd.to_numeric(d[value_col], errors="coerce"),
            }
        ).dropna(subset=[id_col])
    )
    id2value = (
        df.dropna(subset=[value_col])
        .set_index(id_col)[value_col]
        .astype(float)
        .to_dict()
    )
    id2name = (
        df.dropna(subset=[name_col])
        .set_index(id_col)[name_col]
        .astype(str)
        .to_dict()
    )
    return id2value, id2name

def score_to_hex(
    value: Optional[float],
    *,
    cmap: str = "Temps",
    vmin: float = 0.0,
    vmax: float = 1.0,
    na_color: str = "#00000000",
) -> str:
    """
    Map a numeric value to a hex color using a Plotly colorscale.

    Args:
        value(float, optional): Score value to map.
        cmap(str): Plotly colorscale name (e.g. 'Temps', 'Geyser').
        vmin(float): Minimum value for normalization.
        vmax(float): Maximum value for normalization.
        na_color(str): Color for missing values (None/NaN).

    Returns:
        str: Hex color '#RRGGBB' (or na_color for missing values).
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return na_color

    if vmax == vmin:
        t = 1.0
    else:
        t = (float(value) - float(vmin)) / (float(vmax) - float(vmin))
        t = 0.0 if t < 0 else 1.0 if t > 1 else t

    colorscale = px.colors.get_colorscale(cmap)  
    return sample_colorscale(colorscale, [t], colortype="rgb")[0]

def get_svg_attr(
        el: ET.Element, 
        key: str, 
    ) -> Optional[str]:
    """
    Read an SVG attribute, supporting both plain and namespaced forms.

    Args:
        el(Element): SVG element.
        key(str): Attribute name to look for (e.g. 'structure_id').

    Returns:
        str or None: Attribute value if present.
    """
    if key in el.attrib:
        return el.attrib[key]
    return next((v for k, v in el.attrib.items() if k.endswith("}" + key)), None)

def candidate_ids(
        el: ET.Element,
        ) -> List[int]:
    """
    Get structure IDs for an SVG element, including ancestor IDs from structure_id_path.

    Args:
        el(Element): SVG element.

    Returns:
        list[int]: Candidate IDs ordered from most specific to most general.
    """
    ids: List[int] = []
    sid = get_svg_attr(el, "structure_id") or get_svg_attr(el, "structureId")
    if sid and sid.isdigit():
        ids.append(int(sid))

    sid_path = get_svg_attr(el, "structure_id_path") or get_svg_attr(el, "structureIdPath")
    if sid_path:
        parts = [p for p in sid_path.split("/") if p.isdigit()]
        ids.extend(int(p) for p in reversed(parts))

    seen = set()
    return [x for x in ids if not (x in seen or seen.add(x))]

def choose_value_from_candidates(
    cands: List[int],
    id2value: Dict[int, float],
) -> Tuple[Optional[int], Optional[float]]:
    """
    Given a list of candidate Allen IDs (most specific first),
    return the first (id, value) found in id2value.

    Args:
        cands(list[int]): Candidate structure IDs.
        id2value(dict[int, float]): structure_id -> score value.

    Returns:
        Tuple[Optional[int], Optional[float]]: (structure_id, score value) or 
    """
    for sid in cands:
        if sid in id2value:
            try:
                return sid, float(id2value[sid])
            except Exception:
                return sid, None
    return None, None

def _apply_fill_and_stroke(
    el: ET.Element,
    *,
    fill: str,
    fill_opacity: float,
    stroke: str,
    stroke_width: float,
) -> None:
    """
    Apply fill and stroke attributes to an SVG element.

    Args:
        el(Element): SVG element to modify.
        fill(str): Fill color (e.g. '#RRGGBB' or 'none').
        fill_opacity(float): Fill opacity (0.0 to 1.0).
        stroke(str): Stroke color (e.g. '#RRGGBB' or 'none').
        stroke_width(float): Stroke width in pixels.    

    Returns:
        None
    """
    el.set("fill", fill)
    el.set("fill-opacity", str(fill_opacity))
    el.set("stroke", stroke)
    el.set("stroke-width", str(stroke_width))
    # Write style too (SVG often uses style precedence)
    el.set(
        "style",
        (
            f"fill:{fill};"
            f"fill-opacity:{fill_opacity};"
            f"stroke:{stroke};"
            f"stroke-width:{stroke_width}px;"
            "stroke-linejoin:round;"
            "stroke-linecap:round;"
        ),
    )

def recolor_svg_text(
    svg_text: str,
    id2value: Dict[int, float],
    *,
    cmap: str = "Temps",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    stroke: str = "#808080",
    stroke_width: float = 0.6,
    opacity: float = 1.0,
    na_fill: str = "none",
) -> str:
    """
    Recolor an Allen Brain Atlas SVG text by region-level scores.

    Args:
        svg_text(str): Original SVG text.
        id2value(dict[int, float]): structure_id -> score value.
        cmap(str): Plotly colorscale name.
        vmin(float, optional): Min for color normalization (default inferred).
        vmax(float, optional): Max for color normalization (default inferred).
        stroke(str): Stroke color for shapes (e.g. '#RRGGBB' or 'none').
        stroke_width(float): Stroke width in pixels.
        opacity(float): Fill opacity for colored shapes (0.0 to 1.0).
        na_fill(str): Fill color for missing values (e.g. 'none' or '#00000000').

    Returns:
        str: Recolored SVG text.
    """
    if not id2value:
        return svg_text

    vals = [float(v) for v in id2value.values()]
    vmin = min(vals) if vmin is None else float(vmin)
    vmax = max(vals) if vmax is None else float(vmax)

    root = ET.fromstring(svg_text)

    
    matched = 0
    painted = 0
    fills_applied = []
    example_rows = 0

    for el in root.iter():
        tag = el.tag.split("}")[-1].lower()
        if tag not in {"path", "polygon", "polyline"}:
            continue

        cands = candidate_ids(el)  # your existing function (self -> ancestors)
        sid_used, val = choose_value_from_candidates(cands, id2value)

        if sid_used is None:
            continue

        matched += 1

        if val == 0.0:
            fill = na_fill
            fill_opacity = 0.0 if na_fill == "none" else 1.0
        else:
            fill = score_to_hex(val, cmap=cmap, vmin=vmin, vmax=vmax, na_color=na_fill)
            fill_opacity = opacity

        _apply_fill_and_stroke(
            el,
            fill=fill,
            fill_opacity=fill_opacity,
            stroke=stroke,
            stroke_width=stroke_width,
        )

        painted += 1
        fills_applied.append(fill)
    return ET.tostring(root, encoding="unicode")

def recolor_section_svg(
    section_image_id: int,
    out_path: str,
    id2value: Dict[int, float],
    *,
    group_id: int = DEFAULT_GROUP_ID,
    cache: bool = True,
    overwrite: bool = False,
    timeout: int = 60,
    cmap: str = "Temps",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    encoding: str = "utf-8",
) -> str:
    """
    Download an Allen section SVG, recolor it by score, and save it.

    Args:
        section_image_id(int): Allen section image id (sub_images.id).
        out_path(str): Where to save the recolored SVG.
        id2value(dict[int,float]): structure_id -> score value.
        group_id(int): Boundary group id (28 = structure boundaries).
        cache(bool): If True and out_path exists, skip recomputing unless overwrite=True.
        overwrite(bool): If True, overwrite existing out_path.
        timeout(int): Request timeout (seconds).
        cmap(str): Plotly colorscale name.
        vmin(float, optional): Min for color normalization (default inferred).
        vmax(float, optional): Max for color normalization (default inferred).
        encoding(str): Encoding used to decode SVG bytes.

    Returns:
        str: Path to the saved recolored SVG.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if cache and not overwrite and os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    svg_bytes = download_section_svg(
        section_image_id=section_image_id,
        group_id=group_id,
        timeout=timeout,
    )
    svg_text = svg_bytes.decode(encoding, errors="replace")

    recolored = recolor_svg_text(
        svg_text,
        id2value,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    with open(out_path, "w", encoding=encoding) as f:
        f.write(recolored)

    return out_path