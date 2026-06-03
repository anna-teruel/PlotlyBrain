"""
@author Anna Teruel-Sanchis, Jun 2026
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

def load_score(
    score_csv: str,
    id_col: str = "Region ID",
    name_col: str = "Region name",
    value_col: str = "frequency",
) -> pd.DataFrame:
    """
    Load a saved region-level score table.

    Args:
        score_csv : str
            Path to a CSV file produced by save_scores().
        id_col : str, default="Region ID"
            Column containing Allen structure IDs.
        name_col : str, default="Region name"
            Column containing region names.
        value_col : str, default="frequency"
            Score column to visualize.

    Returns:
        pd.DataFrame
            Cleaned score dataframe containing id_col, name_col, and value_col.
    """
    score_df = pd.read_csv(score_csv)

    required = [id_col, name_col, value_col]
    missing = [col for col in required if col not in score_df.columns]

    if missing:
        raise KeyError(
            f"Missing required column(s): {missing}. "
            f"Available columns: {list(score_df.columns)}"
        )

    score_df = score_df[required].copy()
    score_df[id_col] = pd.to_numeric(
        score_df[id_col],
        errors="coerce",
    ).astype("Int64")
    score_df[value_col] = pd.to_numeric(
        score_df[value_col],
        errors="coerce",
    )
    score_df = score_df.dropna(subset=[id_col])

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

