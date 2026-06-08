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

