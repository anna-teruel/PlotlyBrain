"""
Recolor Allen Brain Atlas sections SVGs using region-level scores
@author @anna-teruel, Jan 2026, modified by @KonradDanielewski
"""

import math
import os
import re
import xml.etree.ElementTree as ET
from typing import NoReturn

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.colors import sample_colorscale

from .allen_api import download_section_svg


# Optional deps for better label placement (recommended)
try:
	from svgpathtools import parse_path
	from svgpathtools import Path as SVGPath
except Exception:
	parse_path = None
	SVGPath = None

try:
	from shapely.geometry import Polygon
	from shapely.ops import unary_union
except Exception:
	Polygon = None
	unary_union = None

# Shapely 2.x polylabel (best "center" point)
try:
	from shapely.ops import polylabel as _polylabel
except Exception:
	_polylabel = None


def load_score(
	score_csv: str,
	id_col: str = "Region ID",
	name_col: str = "Region name",
	value_col: str = "frequency",
) -> tuple[dict[int, float], dict[int, str]]:
	"""
	Load a region-level score CSV and build lookup dictionaries.

	Args:
	    score_csv: Path to a CSV file produced by scores.py.
	    id_col: Column name containing Allen structure IDs.
	    name_col: Column name containing region names.
	    value_col: Column name containing the score to color by
	               (e.g. 'frequency' or 'relative_abundance_z').

	Returns:
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
		)
		.dropna(subset=[id_col])
	)
	id2value = df.dropna(subset=[value_col]).set_index(id_col)[value_col].astype(float).to_dict()
	id2name = df.dropna(subset=[name_col]).set_index(id_col)[name_col].astype(str).to_dict()
	return id2value, id2name


def score_to_hex(
	value: float | None,
	*,
	cmap: str = "Temps",
	vmin: float = 0.0,
	vmax: float = 1.0,
	na_color: str = "#00000000",
) -> str:
	"""
	Map a numeric value to a hex/rgb color using a Plotly colorscale.

	Args:
	    value: Score value to map.
	    cmap: Plotly colorscale name (e.g. 'Temps', 'Geyser', 'Sunset').
	    vmin: Minimum value for normalization.
	    vmax: Maximum value for normalization.
	    na_color: Color for missing values (None/NaN).

	Returns:
	    str: Color string (Plotly returns rgb(...)) or na_color.
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
) -> str | None:
	"""
	Read an SVG attribute, supporting both plain and namespaced forms.

	Args:
	    el: SVG element.
	    key: Attribute name to look for (e.g. 'structure_id').

	Returns:
	    Attribute value if present.
	"""
	if key in el.attrib:
		return el.attrib[key]
	return next((v for k, v in el.attrib.items() if k.endswith("}" + key)), None)


def candidate_ids(
	el: ET.Element,
) -> list[int]:
	"""
	Get structure IDs for an SVG element, including ancestor IDs from structure_id_path.

	Args:
	    el: SVG element.

	Returns:
	    list[int]: Candidate IDs ordered from most specific to most general.
	"""
	ids: list[int] = []
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
	cands: list[int],
	id2value: dict[int, float],
) -> tuple[int | None, float | None]:
	"""
	Given a list of candidate Allen IDs (most specific first),
	return the first (id, value) found in id2value.

	Args:
	    cands: Candidate structure IDs.
	    id2value: structure_id -> score value.

	Returns:
	    (structure_id, score value) or (None, None)
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
) -> NoReturn:
	"""
	Apply fill and stroke attributes to an SVG element.

	Args:
	    el: SVG element to modify.
	    fill: Fill color (e.g. 'rgb(...)' or 'none').
	    fill_opacity: Fill opacity (0.0 to 1.0).
	    stroke: Stroke color (e.g. '#RRGGBB' or 'none').
	    stroke_width: Stroke width in pixels.
	"""
	el.set("fill", fill)
	el.set("fill-opacity", str(fill_opacity))
	el.set("stroke", stroke)
	el.set("stroke-width", str(stroke_width))
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


def _get_svg_size(
	root: ET.Element,
) -> tuple[float, float]:
	"""
	Infer canvas size from viewBox or width/height.

	Args:
	    root: SVG root element.

	Returns:
	    (width, height)
	"""
	vb = root.attrib.get("viewBox")
	if vb:
		parts = [p for p in vb.replace(",", " ").split() if p]
		if len(parts) == 4:
			return float(parts[2]), float(parts[3])

	def _to_float(s: str | None, default: float) -> float:
		if not s:
			return default
		s = s.strip().replace("px", "")
		try:
			return float(s)
		except Exception:
			return default

	w = _to_float(root.attrib.get("width"), 1000.0)
	h = _to_float(root.attrib.get("height"), 1000.0)
	return w, h


def _build_parent_map(root: ET.Element) -> dict[ET.Element, ET.Element]:
	"""
	Build a child->parent mapping for ET elements.

	Args:
	    root: SVG root element.

	Returns:
	    dict mapping each child element to its parent.
	"""
	return {child: parent for parent in root.iter() for child in parent}


def _sample_path_to_points(path, n: int = 900) -> list[tuple[float, float]]:
	"""
	Sample points along an svgpathtools Path.

	Args:
	    path: svgpathtools Path
	    n: number of samples

	Returns:
	    list[(x,y)]
	"""
	L = path.length(error=1e-4)
	if L == 0:
		return []
	ts = np.linspace(0, 1, n, endpoint=False)
	pts = [path.point(t) for t in ts]
	return [(p.real, p.imag) for p in pts]


def _continuous_subpaths(path):
	"""
	Split into continuous subpaths (handles 'M' breaks).
	"""
	if SVGPath is None:
		return []
	sub = []
	current = []
	last_end = None
	for seg in path:
		if (last_end is not None) and (seg.start != last_end):
			if current:
				sub.append(current)
			current = [seg]
		else:
			current.append(seg)
		last_end = seg.end
	if current:
		sub.append(current)
	return [SVGPath(*segs) for segs in sub]


def _path_to_polygons(el: ET.Element, min_points: int = 40):
	"""
	Turn a <path d=...> element into one or more shapely Polygons by sampling.

	Returns:
	    list[Polygon]
	"""
	if parse_path is None or Polygon is None:
		return []
	d = el.attrib.get("d")
	if not d:
		return []
	try:
		path = parse_path(d)
	except Exception:
		return []
	subpaths = _continuous_subpaths(path)
	polys = []
	for sp in subpaths:
		pts = _sample_path_to_points(sp, n=max(min_points, 900))
		if len(pts) < 3:
			continue
		if pts[0] != pts[-1]:
			pts = pts + [pts[0]]
		poly = Polygon(pts)
		if not poly.is_valid:
			poly = poly.buffer(0)
		if poly.is_empty:
			continue
		polys.append(poly)
	return polys


def _label_point_local(el: ET.Element) -> tuple[float, float] | None:
	"""
	Compute a good label point (local coords) for an SVG region shape.

	Strategy:
	    1) polylabel (visual center, inside) if available
	    2) representative_point (inside)
	    3) centroid
	    4) bbox midpoint fallback

	Returns:
	    (x,y) in element's local coordinate space, or None.
	"""
	tag = el.tag.split("}")[-1].lower()

	# We focus on path-based regions (Allen boundaries are usually paths)
	if tag != "path":
		# Fallback: bbox from any numeric coords we can find (often not great)
		if tag in {"polygon", "polyline"}:
			pts = el.attrib.get("points", "")
			nums = re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", pts)
			if len(nums) >= 4:
				vals = list(map(float, nums))
				xs = vals[0::2]
				ys = vals[1::2]
				if len(xs) >= 2 and len(ys) >= 2:
					return (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
		return None

	polys = _path_to_polygons(el)
	if polys:
		geom = None
		if unary_union is not None:
			try:
				geom = unary_union(polys)
			except Exception:
				geom = None
		if geom is None:
			geom = max(polys, key=lambda p: abs(p.area))

		if geom is not None and (not geom.is_empty):
			if _polylabel is not None:
				try:
					p = _polylabel(geom, tolerance=1.0)
					return float(p.x), float(p.y)
				except Exception:
					pass
			try:
				p = geom.representative_point()
				return float(p.x), float(p.y)
			except Exception:
				c = geom.centroid
				return float(c.x), float(c.y)

	# bbox fallback for path 'd'
	d = el.attrib.get("d", "")
	nums = re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", d)
	if len(nums) >= 4:
		vals = list(map(float, nums))
		xs = vals[0::2]
		ys = vals[1::2]
		if len(xs) >= 2 and len(ys) >= 2:
			return (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0

	return None


def _append_label_same_parent(
	root: ET.Element,
	parent_map: dict[ET.Element, ET.Element],
	el: ET.Element,
	*,
	x: float,
	y: float,
	text: str,
	font_size: int = 10,
	fill: str = "#000000",
	halo: bool = True,
	halo_color: str = "#ffffff",
	halo_width: float = 3.0,
) -> NoReturn:
	"""
	Append a <text> label to the SAME parent as `el` (like your lxml script).

	This avoids coordinate-space issues from nested <g transform="...">.
	Optionally copies element-level transform onto the text.

	Args:
	    root: SVG root.
	    parent_map: child->parent mapping.
	    el: target SVG element (region shape).
	    x,y: local coordinates.
	    text: label text.
	"""
	parent = parent_map.get(el, root)

	# If the shape has its own transform, copy it to the label.
	el_transform = el.attrib.get("transform")

	style_base = (
		f"font-size:{font_size}px;"
		"font-family:Arial;"
		"text-anchor:middle;"
		"dominant-baseline:middle;"
	)

	def _make_text(*, do_halo: bool) -> ET.Element:
		attrib = {
			"x": str(x),
			"y": str(y),
			"style": style_base,
		}
		if el_transform:
			attrib["transform"] = el_transform

		if do_halo:
			attrib["fill"] = "none"
			attrib["stroke"] = halo_color
			attrib["stroke-width"] = str(halo_width)
			attrib["paint-order"] = "stroke"
		else:
			attrib["fill"] = fill

		t = ET.SubElement(parent, "text", attrib)
		t.text = text
		return t

	if halo:
		_make_text(do_halo=True)
	_make_text(do_halo=False)


def _add_colorbar_group(
	root: ET.Element,
	*,
	cmap: str,
	vmin: float,
	vmax: float,
	title: str,
	x: float,
	y: float,
	bar_w: float = 18,
	bar_h: float = 220,
	n_steps: int = 120,
	n_ticks: int = 7,
	font_size: int = 12,
) -> NoReturn:
	"""
	Add a vertical colorbar (many thin rects) with multiple tick labels.

	Args:
	    root: SVG root.
	    cmap: Plotly colorscale name.
	    vmin/vmax: scale min/max.
	    title: label.
	    x/y: top-left of bar.
	    n_ticks: number of labeled ticks.
	"""
	g = ET.SubElement(root, "g", {"id": "plotlybrain_colorbar"})

	step_h = bar_h / n_steps
	for i in range(n_steps):
		t = 1.0 - i / (n_steps - 1)
		val = vmin + t * (vmax - vmin)
		col = score_to_hex(val, cmap=cmap, vmin=vmin, vmax=vmax)
		ET.SubElement(
			g,
			"rect",
			{
				"x": str(x),
				"y": str(y + i * step_h),
				"width": str(bar_w),
				"height": str(step_h + 0.5),
				"fill": col,
				"stroke": "none",
			},
		)

	ET.SubElement(
		g,
		"rect",
		{
			"x": str(x),
			"y": str(y),
			"width": str(bar_w),
			"height": str(bar_h),
			"fill": "none",
			"stroke": "#333333",
			"stroke-width": "0.8",
		},
	)

	ET.SubElement(
		g,
		"text",
		{
			"x": str(x),
			"y": str(y - 6),
			"font-size": str(font_size),
			"fill": "#222222",
		},
	).text = str(title)

	n_ticks = max(2, int(n_ticks))
	for j in range(n_ticks):
		f = j / (n_ticks - 1)  # 0..1
		val = vmin + f * (vmax - vmin)
		yy = y + (1.0 - f) * bar_h

		ET.SubElement(
			g,
			"line",
			{
				"x1": str(x + bar_w),
				"y1": str(yy),
				"x2": str(x + bar_w + 6),
				"y2": str(yy),
				"stroke": "#333333",
				"stroke-width": "0.8",
			},
		)

		ET.SubElement(
			g,
			"text",
			{
				"x": str(x + bar_w + 9),
				"y": str(yy),
				"font-size": str(font_size),
				"fill": "#222222",
				"dominant-baseline": "middle",
			},
		).text = f"{val:.3g}"


def recolor_svg_text(
	svg_text: str,
	id2value: dict[int, float],
	id2name: dict[int, str] | None = None,
	cmap: str = "Temps",
	vmin: float | None = None,
	vmax: float | None = None,
	stroke: str = "#808080",
	stroke_width: float = 0.6,
	opacity: float = 1.0,
	na_fill: str = "none",
	add_colorbar: bool = True,
	colorbar_title: str = "score",
	colorbar_ticks: int = 7,
	add_centroid_labels: bool = True,
	label_font_size: int = 10,
	label_value_fmt: str = ".2f",
	label_max_chars: int = 40,
	label_value: bool = True,
) -> str:
	"""
	Recolor an Allen Brain Atlas SVG text by region-level scores, and optionally:
	- add an in-SVG colorbar with multiple ticks
	- add region labels placed at a visually centered point inside each region

	Args:
	    svg_text: Original SVG text.
	    id2value: structure_id -> score value.
	    id2name: structure_id -> region name (for labels).
	    cmap: Plotly colorscale name.
	    vmin/vmax: color normalization (default inferred).
	    stroke/stroke_width: outline style.
	    opacity: fill opacity for colored shapes.
	    na_fill: fill for missing/zero values ("none" for transparent).
	    add_colorbar: whether to embed colorbar.
	    colorbar_title: colorbar label.
	    colorbar_ticks: number of tick labels.
	    add_centroid_labels: whether to label regions.
	    label_font_size/value_fmt/max_chars: label formatting.
	    label_value: include numeric value in label.

	Returns:
	    Recolored SVG text.
	"""
	if not id2value:
		return svg_text

	vals = [float(v) for v in id2value.values()]
	vmin = min(vals) if vmin is None else float(vmin)
	vmax = max(vals) if vmax is None else float(vmax)

	root = ET.fromstring(svg_text)
	parent_map = _build_parent_map(root)

	labeled: set[int] = set()

	for el in root.iter():
		tag = el.tag.split("}")[-1].lower()
		if tag not in {"path", "polygon", "polyline"}:
			continue

		cands = candidate_ids(el)
		sid_used, val = choose_value_from_candidates(cands, id2value)
		if sid_used is None:
			continue

		# Your original behavior: val==0 -> na_fill (transparent)
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

		# labels (once per sid)
		if add_centroid_labels and id2name is not None and sid_used not in labeled:
			xy = _label_point_local(el)
			if xy is not None:
				name = id2name.get(sid_used, f"ID {sid_used}")
				if len(name) > label_max_chars:
					name = name[: max(0, label_max_chars - 1)] + "…"

				txt = name
				if label_value and (val is not None) and not (isinstance(val, float) and math.isnan(val)):
					txt = f"{name} ({val:{label_value_fmt}})"

				_append_label_same_parent(
					root,
					parent_map,
					el,
					x=float(xy[0]),
					y=float(xy[1]),
					text=txt,
					font_size=label_font_size,
					fill="#000000",
					halo=True,
				)
				labeled.add(sid_used)

	# colorbar
	if add_colorbar:
		w, _h = _get_svg_size(root)
		legend_x = w - 90
		legend_y = 30
		_add_colorbar_group(
			root,
			cmap=cmap,
			vmin=vmin,
			vmax=vmax,
			title=colorbar_title,
			x=legend_x,
			y=legend_y,
			bar_w=18,
			bar_h=220,
			n_steps=140,
			n_ticks=int(colorbar_ticks),
			font_size=12,
		)

	return ET.tostring(root, encoding="unicode")


def recolor_section_svg(
	section_image_id: int,
	out_path: str,
	id2value: dict[int, float],
	id2name: dict[int, str] | None = None,
	group_id: int = 28,
	cache: bool = True,
	overwrite: bool = False,
	timeout: int = 60,
	cmap: str = "Temps",
	vmin: float | None = None,
	vmax: float | None = None,
	encoding: str = "utf-8",
	add_colorbar: bool = True,
	colorbar_title: str = "score",
	colorbar_ticks: int = 7,
	add_centroid_labels: bool = True,
	label_font_size: int = 10,
	label_value_fmt: str = ".2f",
	label_max_chars: int = 40,
	label_value: bool = True,
) -> str:
	"""
	Download an Allen section SVG, recolor it by score, and save it.

	Args:
	    section_image_id: Allen section image id (sub_images.id).
	    out_path: Where to save the recolored SVG.
	    id2value: structure_id -> score value.
	    id2name: structure_id -> region name (for labels).
	    group_id: Boundary group id (28 = structure boundaries).
	    cache: If True and out_path exists, skip recomputing unless overwrite=True.
	    overwrite: If True, overwrite existing out_path.
	    timeout: Request timeout (seconds).
	    cmap: Plotly colorscale name.
	    vmin/vmax: normalization range.
	    encoding: Encoding used to decode SVG bytes.
	    add_colorbar: embed a colorbar.
	    colorbar_title: colorbar label.
	    colorbar_ticks: tick count.
	    add_centroid_labels: add labels.
	    label_*: label formatting.

	Returns:
	    Path to the saved recolored SVG.
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
		id2name=id2name,
		cmap=cmap,
		vmin=vmin,
		vmax=vmax,
		stroke="#808080",
		stroke_width=0.6,
		opacity=1.0,
		na_fill="none",
		add_colorbar=add_colorbar,
		colorbar_title=colorbar_title,
		colorbar_ticks=colorbar_ticks,
		add_centroid_labels=add_centroid_labels,
		label_font_size=label_font_size,
		label_value_fmt=label_value_fmt,
		label_max_chars=label_max_chars,
		label_value=label_value,
	)

	with open(out_path, "w", encoding=encoding) as f:
		f.write(recolored)

	return out_path