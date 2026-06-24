"""
Utilities for loading score tables, GeoJSON files and saving
Plotly figures generated from region-level brain atlas analyses.
"""

import json
import os

import pandas as pd
import plotly.graph_objects as go


def load_score(
	score_csv: str,
	id_col: str = "Region ID",
	name_col: str = "Region name",
) -> pd.DataFrame:
	"""
	Load a region-level score table.

	Args:
	    score_csv : str
	        Path to a score table file.
	    id_col : str, default="Region ID"
	        Column containing Allen structure IDs.
	    name_col : str, default="Region name"
	        Column containing region names.

	Returns:
	    pd.DataFrame
	        Score dataframe containing all saved score columns.
	"""
	score_df = pd.read_csv(score_csv)
	return score_df


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


def save_figure(
	fig: go.Figure,
	out_dir: str,
	filename: str = "figure",
	extension: str = "svg",
) -> str:
	"""
	Save plotly figure to disk.

	Supported formats depend on the output file extension
	and Plotly's image export backend (kaleido).

	Args:
	    fig: go.Figure
	        Plotly figure to save
	    out_dir: str
	        output directory
	     filename : str, default="figure"
	        Figure filename without extension.
	    extension : {"svg", "pdf", "png", "jpg", "html"},
	        default="svg"
	        Output file format.

	Returns:
	    str:
	        Absolute path to saved file.
	"""
	os.makedirs(out_dir, exist_ok=True)

	out_path = os.path.join(
		out_dir,
		f"{filename}.{extension}",
	)

	if extension == "html":
		fig.write_html(out_path)
	else:
		fig.write_image(out_path)
	return os.path.abspath(out_path)
