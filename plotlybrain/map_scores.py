"""
Recolor Allen Brain Atlas sections SVGs using region-level scores
@author @anna-teruel, Jan 2026, modified by @KonradDanielewski
"""
"""
Recolor Allen Brain Atlas sections SVGs using region-level scores
@author Anna Teruel-Sanchis, Jan 2026
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

# Optional deps (recommended for robust centroiding on <path>)
# pip install svgpathtools shapely
try:
	from svgpathtools import parse_path
	from svgpathtools import Path as SVGPath
except Exception:
	parse_path = None
	SVGPath = None

try:
	from shapely.geometry import Polygon
except Exception:
	Polygon = None


# =========================
# CSV loading
# =========================
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


# =========================
# Color mapping
# =========================
def score_to_hex(
	value: float | None,
	*,
	cmap: str = "Temps",
	vmin: float = 0.0,
	vmax: float = 1.0,
	na_color: str = "#00000000",
) -> str:
	"""
	Map a numeric value to a color using a Plotly colorscale.

	Args:
	    value: Score value to map.
	    cmap: Plotly colorscale name (e.g. 'Temps', 'Geyser', 'Sunset').
	    vmin: Minimum value for normalization.
	    vmax: Maximum value for normalization.
	    na_color: Color for missing values (None/NaN).

	Returns:
	    str: Color string (Plotly returns 'rgb(r,g,b)') or na_color.
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


# =========================
# SVG ID handling
# =========================
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


def extract_structure_ids_from_svg_text(svg_text: str) -> list[int]:
	"""
	Extract all structure IDs present in an SVG (from structure_id and structure_id_path).

	Args:
	    svg_text: SVG as text.

	Returns:
	    Sorted list of unique structure IDs.
	"""
	root = ET.fromstring(svg_text)
	ids = set()
	for el in root.iter():
		sid = get_svg_attr(el, "structure_id") or get_svg_attr(el, "structureId")
		if sid and sid.isdigit():
			ids.add(int(sid))
		sid_path = get_svg_attr(el, "structure_id_path") or get_svg_attr(el, "structureIdPath")
		if sid_path:
			for p in sid_path.strip("/").split("/"):
				if p.isdigit():
					ids.add(int(p))
	return sorted(ids)


# =========================
# SVG style helpers
# =========================
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


# =========================
# Transform handling (clean + needed)
# =========================
def _mat_identity() -> list[list[float]]:
	"""3x3 identity matrix."""
	return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _mat_mul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
	"""
	2D affine matrix multiply. Matrices are 3x3.

	This replaces the long "ugly" explicit version with the standard definition:
	(A @ B)[i,j] = sum_k A[i,k] * B[k,j]
	"""
	return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _mat_apply(M: list[list[float]], x: float, y: float) -> tuple[float, float]:
	"""Apply 3x3 affine matrix to (x,y)."""
	xx = M[0][0] * x + M[0][1] * y + M[0][2]
	yy = M[1][0] * x + M[1][1] * y + M[1][2]
	return xx, yy


_transform_cmd = re.compile(r"([a-zA-Z]+)\(([^)]*)\)")


def _parse_transform(transform: str) -> list[list[list[float]]]:
	"""
	Parse SVG transform string into a list of 3x3 matrices (in order).

	Supports:
	    - matrix(a,b,c,d,e,f)
	    - translate(tx[,ty])
	    - scale(sx[,sy])
	    - rotate(angle[,cx,cy])  (common enough to include)

	Returns:
	    list of 3x3 matrices, in the same order as SVG applies them.
	"""
	mats = []
	for name, args in _transform_cmd.findall(transform or ""):
		nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", args)]
		name = name.strip().lower()

		if name == "matrix" and len(nums) == 6:
			a, b, c, d, e, f = nums
			mats.append([[a, c, e], [b, d, f], [0.0, 0.0, 1.0]])

		elif name == "translate" and len(nums) >= 1:
			tx = nums[0]
			ty = nums[1] if len(nums) >= 2 else 0.0
			mats.append([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])

		elif name == "scale" and len(nums) >= 1:
			sx = nums[0]
			sy = nums[1] if len(nums) >= 2 else sx
			mats.append([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])

		elif name == "rotate" and len(nums) >= 1:
			ang = nums[0] * math.pi / 180.0
			ca = math.cos(ang)
			sa = math.sin(ang)

			# rotate about origin
			R = [[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]]

			# optional center (cx, cy): T(cx,cy) * R * T(-cx,-cy)
			if len(nums) >= 3:
				cx, cy = nums[1], nums[2]
				T1 = [[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]]
				T0 = [[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]]
				mats.append(_mat_mul(_mat_mul(T1, R), T0))
			else:
				mats.append(R)

	# (skewX/skewY not included; add if you ever see them)
	return mats


def _compose_transform_to_root(el: ET.Element, parent_map: dict[ET.Element, ET.Element]) -> list[list[float]]:
	"""
	Compose transforms from root->...->el (document order).

	Args:
	    el: element to compute transform for.
	    parent_map: child -> parent mapping.

	Returns:
	    3x3 matrix mapping element-local coords to root coords.
	"""
	chain = []
	cur = el
	while cur is not None:
		chain.append(cur)
		cur = parent_map.get(cur)
	chain = list(reversed(chain))

	M = _mat_identity()
	for node in chain:
		tf = node.attrib.get("transform")
		if tf:
			for m in _parse_transform(tf):
				M = _mat_mul(M, m)
	return M


# =========================
# Robust centroid (local coords)
# =========================
def _sample_path_to_points(path, n: int = 800) -> list[tuple[float, float]]:
	"""Sample N points uniformly along path length (works for curves)."""
	L = path.length(error=1e-4)
	if L == 0:
		return []
	ts = np.linspace(0, 1, n, endpoint=False)
	pts = [path.point(t) for t in ts]
	return [(p.real, p.imag) for p in pts]


def _continuous_subpaths(path):
	"""Split into continuous subpaths (handles 'M' breaks)."""
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

	if SVGPath is None:
		return []
	return [SVGPath(*segs) for segs in sub]


def _path_to_polygons_from_d(d: str, min_points: int = 40):
	"""Turn a <path d=...> into shapely Polygons by sampling."""
	if (parse_path is None) or (Polygon is None) or (not d):
		return []

	try:
		path = parse_path(d)
	except Exception:
		return []

	subpaths = _continuous_subpaths(path)
	polys = []
	for sp in subpaths:
		pts = _sample_path_to_points(sp, n=max(min_points, 800))
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


def _estimate_centroid_local(
	el: ET.Element,
) -> tuple[float, float] | None:
	"""
	Robust centroid in LOCAL element coordinates.

	For <path>:
	    - sample curve into polygons
	    - choose largest polygon
	    - use representative_point() (guaranteed inside)
	Fallback:
	    - bbox midpoint using numbers in 'd'

	For polygon/polyline:
	    - representative_point() if shapely is available
	    - else mean of vertices

	Returns:
	    (cx, cy) in local coords, or None.
	"""
	tag = el.tag.split("}")[-1].lower()

	if tag == "path":
		d = el.attrib.get("d", "")
		polys = _path_to_polygons_from_d(d)
		if polys:
			big = max(polys, key=lambda p: abs(p.area))
			try:
				p = big.representative_point()
				return float(p.x), float(p.y)
			except Exception:
				c = big.centroid
				return float(c.x), float(c.y)

		# fallback bbox midpoint of d numbers
		nums = re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", d)
		if len(nums) >= 4:
			vals = list(map(float, nums))
			xs = vals[0::2]
			ys = vals[1::2]
			if len(xs) >= 2 and len(ys) >= 2:
				return (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
		return None

	if tag in {"polygon", "polyline"}:
		points = el.attrib.get("points", "")
		nums = re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", points)
		if len(nums) < 4:
			return None
		vals = list(map(float, nums))
		if len(vals) % 2 != 0:
			vals = vals[:-1]
		pts = list(zip(vals[0::2], vals[1::2]))
		if len(pts) < 3:
			return None

		if Polygon is not None:
			try:
				poly = Polygon(pts)
				if not poly.is_valid:
					poly = poly.buffer(0)
				if not poly.is_empty:
					p = poly.representative_point()
					return float(p.x), float(p.y)
			except Exception:
				pass

		xs = [p[0] for p in pts]
		ys = [p[1] for p in pts]
		return float(np.mean(xs)), float(np.mean(ys))

	return None


def _append_label_text(
	parent: ET.Element,
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
	Append <text> at (x,y). Use halo for readability.

	IMPORTANT:
	    Here we append to ROOT using ROOT coordinates (after applying transforms).
	"""
	if halo:
		t_halo = ET.SubElement(
			parent,
			"text",
			{
				"x": str(x),
				"y": str(y),
				"fill": "none",
				"stroke": halo_color,
				"stroke-width": str(halo_width),
				"paint-order": "stroke",
				"font-size": str(font_size),
				"font-family": "Arial",
				"text-anchor": "middle",
				"dominant-baseline": "middle",
			},
		)
		t_halo.text = text

	t = ET.SubElement(
		parent,
		"text",
		{
			"x": str(x),
			"y": str(y),
			"fill": fill,
			"font-size": str(font_size),
			"font-family": "Arial",
			"text-anchor": "middle",
			"dominant-baseline": "middle",
		},
	)
	t.text = text


# =========================
# Colorbar overlay (ticks > min/max)
# =========================
def _add_colorbar_group(
	root: ET.Element,
	*,
	cmap: str,
	vmin: float,
	vmax: float,
	x: float,
	y: float,
	bar_w: float = 18,
	bar_h: float = 220,
	n_steps: int = 140,
	n_ticks: int = 7,
	font_size: int = 12,
	title: str = "score",
) -> NoReturn:
	"""
	Add a vertical colorbar (approximated with many thin rects) with multiple tick labels.

	Args:
	    root: SVG root.
	    cmap: Plotly colorscale name.
	    vmin/vmax: scale min/max.
	    x/y: position.
	    bar_w/bar_h: bar size.
	    n_steps: gradient resolution.
	    n_ticks: number of labeled ticks (>=2).
	    title: colorbar title.
	"""
	g = ET.SubElement(root, "g", {"id": "plotlybrain_colorbar"})

	step_h = bar_h / n_steps
	for i in range(n_steps):
		t = 1.0 - i / (n_steps - 1)  # top=vmax
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
		tt = j / (n_ticks - 1)  # 0 bottom .. 1 top
		val = vmin + tt * (vmax - vmin)
		yy = y + bar_h * (1.0 - tt)

		ET.SubElement(
			g,
			"line",
			{
				"x1": str(x + bar_w),
				"y1": str(yy),
				"x2": str(x + bar_w + 5),
				"y2": str(yy),
				"stroke": "#333333",
				"stroke-width": "0.8",
			},
		)

		ET.SubElement(
			g,
			"text",
			{
				"x": str(x + bar_w + 8),
				"y": str(yy),
				"font-size": str(font_size),
				"fill": "#222222",
				"dominant-baseline": "middle",
			},
		).text = f"{val:.3g}"


# =========================
# Main recoloring function
# =========================
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
	label_value: bool = True,
	label_font_size: int = 9,
	label_value_fmt: str = ".2f",
	label_max_chars: int = 40,
) -> str:
	"""
	Recolor an Allen Brain Atlas SVG text by region-level scores.
	Optionally embed a colorbar and place centroid labels on each painted region.

	IMPORTANT:
	    Labels are placed using root coordinates after applying the full transform chain,
	    so they overlay correctly even when the SVG contains nested transforms.

	Args:
	    svg_text: Original SVG text.
	    id2value: structure_id -> score value.
	    id2name: structure_id -> region name (optional; required for labels).
	    cmap: Plotly colorscale name.
	    vmin/vmax: color normalization range (default inferred).
	    stroke/stroke_width: outline.
	    opacity: fill opacity.
	    na_fill: fill for val==0 or missing, typically "none".
	    add_colorbar: embed a colorbar.
	    colorbar_title: label above the colorbar.
	    colorbar_ticks: number of tick labels.
	    add_centroid_labels: add region name labels.
	    label_value: include the numeric value in label.
	    label_font_size/value_fmt/max_chars: label formatting.

	Returns:
	    Recolored SVG text.
	"""
	if not id2value:
		return svg_text

	vals = [float(v) for v in id2value.values()]
	vmin = min(vals) if vmin is None else float(vmin)
	vmax = max(vals) if vmax is None else float(vmax)

	root = ET.fromstring(svg_text)
	parent_map = {child: parent for parent in root.iter() for child in parent}

	labeled_sids: set[int] = set()

	for el in root.iter():
		tag = el.tag.split("}")[-1].lower()
		if tag not in {"path", "polygon", "polyline"}:
			continue

		cands = candidate_ids(el)
		sid_used, val = choose_value_from_candidates(cands, id2value)
		if sid_used is None:
			continue

		# fill logic (you can change this if you want 0.0 to be colored instead of blank)
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

		# centroid labels (once per region id used)
		if add_centroid_labels and (id2name is not None) and (sid_used not in labeled_sids):
			xy_local = _estimate_centroid_local(el)
			if xy_local is None:
				continue

			# local -> root coords (THIS is what makes overlays correct)
			M = _compose_transform_to_root(el, parent_map)
			xx, yy = _mat_apply(M, float(xy_local[0]), float(xy_local[1]))

			name = id2name.get(sid_used, f"ID {sid_used}")
			if len(name) > label_max_chars:
				name = name[: max(0, label_max_chars - 1)] + "…"

			if label_value and (val is not None) and not (isinstance(val, float) and math.isnan(val)):
				label_txt = f"{name} ({val:{label_value_fmt}})"
			else:
				label_txt = name

			_append_label_text(
				root,
				x=float(xx),
				y=float(yy),
				text=label_txt,
				font_size=label_font_size,
				fill="#000000",
				halo=True,
				halo_color="#ffffff",
				halo_width=3.0,
			)
			labeled_sids.add(sid_used)

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
			x=legend_x,
			y=legend_y,
			bar_w=18,
			bar_h=220,
			n_steps=140,
			n_ticks=int(colorbar_ticks),
			font_size=12,
			title=colorbar_title,
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
	label_value: bool = True,
	label_font_size: int = 9,
	label_value_fmt: str = ".2f",
	label_max_chars: int = 40,
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
	    vmin/vmax: Min/Max for color normalization.
	    encoding: SVG decode encoding.

	Returns:
	    Path to saved recolored SVG.
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
		label_value=label_value,
		label_font_size=label_font_size,
		label_value_fmt=label_value_fmt,
		label_max_chars=label_max_chars,
	)

	with open(out_path, "w", encoding=encoding) as f:
		f.write(recolored)

	return out_path