"""
Recolor Allen Brain Atlas sections SVGs using region-level scores
@author Anna Teruel-Sanchis, Jan 2026
"""

from __future__ import annotations

import os
import re
import math
from typing import Dict, Optional, Tuple, List, Iterable, Union
from dataclasses import dataclass
import pandas as pd
import plotly.express as px
from plotly.colors import sample_colorscale
import xml.etree.ElementTree as ET

from .allen_api import (
    ALLEN_API_BASE,
    DEFAULT_ATLAS_ID,
    DEFAULT_GROUP_ID,
    fetch_section_image_ids,
    download_section_svg,
    AllenAPIError,
)

def load_score(
    score_hdf5: str,
    *,
    id_col: str = "Region ID",
    name_col: str = "Region name",
    value_col: str = "frequency",
) -> Tuple[Dict[int, float], Dict[int, str]]:
    """
    Load a region-level score hdf5 and build lookup dictionaries.

    Args:
        score_hdf5(str): Path to an HDF5 file produced by scores.py.
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
        pd.read_hdf(score_hdf5, key="scores")[[id_col, name_col, value_col]]
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
    return sample_colorscale(colorscale, t, colortype="hex")[0]

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


def recolor_svg_text(
        svg_text: str, 
        id2value: Dict[int, float], 
        *, 
        cmap="Temps",
        vmin=None, 
        vmax=None, 
        stroke="#444444", 
        stroke_width=0.3,
        opacity=1.0, 
        na_fill="#00000000", 
        ) -> str:
    """
    Recolor an Allen Brain Atlas SVG-text section using region-level scores.
    Color values are obtained by mapping numeric scores to a Plotly colorscale.
    Regions without an associated score are rendered transparent.

    Args:
        svg_text (str): Raw SVG content of an Allen atlas section.
        id2value (dict[int, float]): Mapping from Allen structure IDs to scores.
        cmap (str): Plotly colorscale name used for coloring.
        vmin (float, optional): Minimum score used for color normalization.
                               If None, inferred from the data.
        vmax (float, optional): Maximum score used for color normalization.
                               If None, inferred from the data.
        stroke (str): Color used for region outlines.
        stroke_width (float): Width of region outlines.
        opacity (float): Fill opacity for scored regions.
        na_fill (str): Fill color for regions without a score
                       (default: transparent).

    Returns:
        str: SVG text with anatomical regions recolored according to the scores.
    """
    if not id2value:
        return svg_text

    vals = [float(v) for v in id2value.values()]
    vmin = min(vals) if vmin is None else float(vmin)
    vmax = max(vals) if vmax is None else float(vmax)

    root = ET.fromstring(svg_text)

    for el in root.iter():
        tag = el.tag.split("}")[-1].lower()
        if tag not in {"path", "polygon", "polyline"}:
            continue
        val = None
        for sid in candidate_ids(el):
            if sid in id2value:
                val = id2value[sid]
                break

        fill = score_to_hex(val, cmap=cmap, vmin=vmin, vmax=vmax, na_color=na_fill)
        el.set("fill", fill)  #color inside the shape
        el.set("fill-opacity", str(opacity) if fill != na_fill else "0") #transparency of the shape
        el.set("stroke", stroke) #color of the outline
        el.set("stroke-width", str(stroke_width)) #thickness of the outline

    return ET.tostring(root, encoding="unicode")

def recolor_section_svg(
    section_image_id: int,
    out_path: str,
    id2value: Dict[int, float],
    *,
    svg_dir: str = "allen_svgs",
    group_id: int = DEFAULT_GROUP_ID,
    cache: bool = True,
    overwrite: bool = False,
    timeout: int = 60,
    cmap: str = "Temps",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> str:
    """
    Download (if needed) and recolor one Allen section SVG, then save it.

    Args:
        section_image_id(int): Allen section image (sub_image) ID.
        out_path(str): Output path for recolored SVG.
        id2value(dict[int, float]): structure_id -> score value.
        svg_dir(str): Directory used to cache raw SVG downloads.
        group_id(int): Allen SVG boundary group ID (default 28).
        cache(bool): Reuse cached raw SVG file if present.
        overwrite(bool): Overwrite existing recolored SVG.
        timeout(int): Request timeout in seconds.
        cmap(str): Plotly colorscale name.
        vmin(float, optional): Normalization min.
        vmax(float, optional): Normalization max.

    Returns:
        str: Path to the saved recolored SVG.
    """
    os.makedirs(svg_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    raw_svg = os.path.join(svg_dir, f"{section_image_id}.svg")
    download_section_svg(
        section_image_id=section_image_id,
        out_path=raw_svg,
        group_id=group_id,
        cache=cache,
        overwrite=overwrite,
        timeout=timeout,
    )

    if (not overwrite) and os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    with open(raw_svg, "r", encoding="utf-8") as f:
        svg_text = f.read()
    svg_text = recolor_svg_text(svg_text, id2value, cmap=cmap, vmin=vmin, vmax=vmax)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(svg_text)

    return out_path