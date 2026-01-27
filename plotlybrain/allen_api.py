"""
Utilities for downloading anatomical SVG templates from the Allen Mouse Brain Atlas.
Data acquisition. 
@author Anna Teruel-Sanchis, Jan 2026
"""

from __future__ import annotations
import os
import time
from typing import List
import requests

ALLEN_API_BASE = "https://api.brain-map.org/api/v2"
DEFAULT_ATLAS_ID = 1 
DEFAULT_GROUP_ID = 28  

class AllenAPIError(RuntimeError):
    """Raised when Allen API requests fail or return unexpected content."""

def fetch_section_image_ids(
        atlas_id: int = DEFAULT_ATLAS_ID,
        include_failed: bool = False,
        timeout: int = 60,
        )-> List[int]:
    """
    Fetch section-image (sub_image) IDs for a given atlas dataset.

    In the Allen Brain Atlas, each atlas (e.g. adult mouse coronal) consists of
    many 2D section images ("sub_images"), each corresponding to one brain slice.
    This function queries the Allen API and returns the list of those section-image
    IDs, which are later used to download SVG boundary files. 

    Args:
        atlas_id(int): Allen atlas dataset ID. Defaults is 1 (adult mouse coronal).
                        For adult mouse sagittal sections set ID to 2. 
        include_failed(bool): If False, tries to exclude failed datasets. These are Allen Datasets that failed quality controls. 
                        This flag protects you from downloading broken or unusable atlas sections.
        timeout(int): Request timeout in seconds. If the Allen server does not respond within timeout seconds, stop waiting.

    Returns:
        List[int]: Sorted list of unique section-image IDs.

    Raises:
        AllenAPIError: If the request fails or no IDs can be parsed.
    """
    criteria = [
        "model::SectionDataSet",
        "rma::criteria",
        f"[atlas_data_set_id$eq{atlas_id}]",
    ]
    if not include_failed:
        criteria.append("[failed$eqfalse]")

    url = (
        f"{ALLEN_API_BASE}/data/query.csv?"
        f"criteria={','.join(criteria)},rma::include,sub_images"
    )

    try:
        txt = requests.get(url, timeout=timeout).text
    except Exception as e:
        raise AllenAPIError(f"Failed to fetch section IDs: {e}") from e

    ids = sorted({
        int(line.split(",")[0])
        for line in txt.splitlines()
        if line.strip() and line.lower() not in ("id", "sub_images.id", "sub_image.id")
        and line.split(",")[0].isdigit()
    })

    if not ids:
        raise AllenAPIError(
            "No section-image IDs parsed. The Allen query may have changed.\n"
            f"URL: {url}\nPreview:\n" + "\n".join(txt.splitlines()[:15])
        )

    return ids

def download_section_svg(
        section_image_id: int,
        out_path: str,
        group_id: int = DEFAULT_GROUP_ID,
        cache: bool = True,
        overwrite: bool = False,
        timeout: int = 60,
        )-> str:
    """
    Download and save the SVG structure-boundary file for a single Allen atlas section.

    Each section image corresponds to one 2D brain slice in the selected atlas.
    This function retrieves the SVG containing anatomical boundaries and saves it
    to disk for later recoloring or visualization.

    Args:
        section_image_id (int): Allen section image (sub_image) ID.
        out_path (str): File path where the SVG will be saved.
        group_id (int): Boundary group ID defining which structures are included
                       (default: 28, structure boundaries). 
        cache (bool): If True, reuse an existing SVG file if present.
        overwrite (bool):If True, force re-download even if the file already exists.
        timeout (int): Maximum time (seconds) to wait for the Allen API response.

    Returns:
        str: path to the saved SVG file.

    Raises:
        AllenAPIError: If the SVG download fails.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if cache and not overwrite and os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    url = f"{ALLEN_API_BASE}/svg_download/{section_image_id}?groups={group_id}"

    try:
        svg_text = requests.get(url, timeout=timeout).text
    except Exception as e:
        raise AllenAPIError(f"Failed to download SVG for section {section_image_id}") from e

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(svg_text)

    return out_path

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