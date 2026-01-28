"""
Utilities for downloading anatomical SVG templates from the Allen Mouse Brain Atlas.
Data acquisition. 
@author Anna Teruel-Sanchis, Jan 2026
"""

from __future__ import annotations
import os
import time
import pandas as pd
from typing import List
import requests
import io

ALLEN_API_BASE = "https://api.brain-map.org/api/v2"
DEFAULT_ATLAS_ID = 1 
DEFAULT_GROUP_ID = 28  

class AllenAPIError(RuntimeError):
    """Raised when Allen API requests fail or return unexpected content."""

def fetch_section_image_ids(
    atlas_id: int = DEFAULT_ATLAS_ID,
    *,
    group_id: int = DEFAULT_GROUP_ID,
    timeout: int = 60,
) -> List[int]:
    """
    Fetch Allen atlas section IDs usable in /svg_download/<id>.

    This reproduces the working notebook query:
    model::AtlasImage -> tabular sub_images.id

    Args:
        atlas_id(int): Atlas ID (1=Mouse P56 Coronal, 2=Mouse P56 Sagittal).
        group_id(int): Graphic group ID (28=structure boundaries).
        timeout(int): Request timeout in seconds.

    Returns:
        List[int]: List of section IDs (sub_images.id).
    """
    url = (
        f"{ALLEN_API_BASE}/data/query.csv?"
        "criteria=model::AtlasImage,"
        f"rma::criteria,atlas_data_set(atlases[id$eq{int(atlas_id)}]),"
        f"graphic_objects(graphic_group_label[id$eq{int(group_id)}]),"
        "rma::options[tabular$eq'sub_images.id'][order$eq'sub_images.id']"
        "&num_rows=all&start_row=0"
    )

    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
    except Exception as e:
        raise AllenAPIError(f"Failed to fetch section IDs: {e}\nURL: {url}") from e

    df = pd.read_csv(io.StringIO(r.text))
    if df.empty:
        raise AllenAPIError(f"Empty response for section IDs.\nURL: {url}\nPreview:\n{r.text[:500]}")

    if "sub_images.id" in df.columns:
        col = "sub_images.id"
    else:
        int_cols = [c for c in df.columns if pd.api.types.is_integer_dtype(df[c])]
        if not int_cols:
            raise AllenAPIError(f"Could not find an integer ID column.\nColumns: {list(df.columns)}")
        col = int_cols[0]

    ids = df[col].dropna().astype(int).tolist()
    if not ids:
        raise AllenAPIError(f"No IDs parsed.\nURL: {url}\nPreview:\n{r.text[:500]}")

    return ids

def download_section_svg(
    section_image_id: int,
    *,
    group_id: int = DEFAULT_GROUP_ID,
    timeout: int = 60,
    ) -> bytes:
    """
    Download and save the SVG structure-boundary file for a single Allen atlas section.

    Each section image corresponds to one 2D brain slice in the selected atlas.
    This function retrieves the SVG containing anatomical boundaries and saves it
    to disk for later recoloring or visualization.

    Args:
        section_image_id (int): Allen section image (sub_image) ID.
        group_id (int): Boundary group ID defining which structures are included
                       (default: 28, structure boundaries). 
        timeout (int): Maximum time (seconds) to wait for the Allen API response.

    Returns:
        str: path to the saved SVG file.

    Raises:
        AllenAPIError: If the SVG download fails.
    """
    url = f"{ALLEN_API_BASE}/svg_download/{int(section_image_id)}?groups={int(group_id)}"
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
    except Exception as e:
        raise AllenAPIError(f"Failed to download SVG: {e}\nURL: {url}") from e
    return r.content

def download_all_svgs(
        out_dir: str,              
        atlas_id: int = DEFAULT_ATLAS_ID,
        group_id: int = DEFAULT_GROUP_ID,
        include_failed: bool = False,
        cache: bool = True,
        overwrite: bool = False,
        timeout: int = 60,
        sleep_s: float = 0.0,
        filename_fmt: str = "{section_id}.svg"):
    """
    Download raw boundary SVGs for all section images in an atlas dataset.

    Each Allen Brain Atlas dataset consists of multiple 2D section images (brain slices).
    For each slice, this function downloads the corresponding SVG file containing
    anatomical structure boundaries (graphic group 28 by default) and saves it to disk.
    These SVGs serve as the anatomical templates that can later be recolored by
    region-level values (e.g., frequency, z-score, or density).

    Args:
        out_dir: Directory to save SVGs.
        atlas_id: Allen atlas dataset ID.
        group_id: Boundary group ID.
        include_failed: Whether to include failed datasets.
        cache: Skip existing files.
        overwrite: Re-download existing files.
        timeout: Request timeout in seconds.
        sleep_s: Optional delay between downloads.
        filename_fmt: Filename pattern containing "{section_id}".
    """
    os.makedirs(out_dir, exist_ok=True)
    ids = fetch_section_image_ids(atlas_id=atlas_id, include_failed=include_failed, timeout=timeout)

    saved: List[str] = []
    for sid in ids:
        out_path = os.path.join(out_dir, filename_fmt.format(section_id=sid))
        saved.append(
            download_section_svg(
                section_image_id=sid,
                out_path=out_path,
                group_id=group_id,
                cache=cache,
                overwrite=overwrite,
                timeout=timeout,
            )
        )
        if sleep_s:
            time.sleep(sleep_s)

    return saved