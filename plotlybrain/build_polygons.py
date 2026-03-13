"""
Build selected per-slice GeoJSON polygons from the Allen CCF annotation volume.
@author @anna-teruel, Mar 2026
"""

import io
import json
import os
from dataclasses import dataclass
from typing import Literal

import nrrd
import numpy as np
import pandas as pd
import requests
from skimage import measure
from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.ops import unary_union

from plotlybrain.coord_system import (
    ap_mm_to_slice_index,
    ap_range_mm_to_slice_indices,
    slice_index_to_ap_mm,
)

# Allen annotation volumes at different isotropic resolutions (microns)
ANNOTATION_URLS = {
    10: "https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_10.nrrd",
    25: "https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_25.nrrd",
    50: "https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_50.nrrd",
    100: "https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_100.nrrd",
}

# Allen ontology tree (Mouse Brain Atlas structure graph 1)
STRUCTURE_GRAPH_URL = "https://api.brain-map.org/api/v2/structure_graph_download/1.json"


@dataclass
class BuildConfig:
    """
    Configuration for building selected slice GeoJSONs from an Allen
    annotation volume.

    Args:
        out_dir : str
            Output directory where GeoJSON slices and optional cached atlas
            files are saved.
        resolution_um : int, default=25
            Annotation volume resolution in microns.
        orientation : {"coronal", "sagittal", "horizontal"}, default="coronal"
            Slice orientation used to extract 2D views from the 3D annotation
            volume.
        ap_values_mm : list[float] | None, default=None
            AP coordinates relative to bregma (in mm) to convert into slice
            indices.
        ap_range_mm : tuple[float, float] | None, default=None
            Inclusive AP interval relative to bregma (in mm) to convert into
            slice indices.
        min_area_px : float, default=5.0
            Minimum polygon/component area in pixels to keep after converting
            a label mask into geometry.
        simplify_px : float, default=0.8
            Polygon simplification tolerance in pixels. Higher values reduce
            vertex count and file size.
        storage_mode : {"disk", "memory"}, default="disk"
            Whether to cache downloaded atlas files to disk or load them only
            in RAM.
        overwrite : bool, default=False
            Whether to overwrite existing cached files in disk mode.
    """
    out_dir: str
    resolution_um: int = 25
    orientation: Literal["coronal", "sagittal", "horizontal"] = "coronal"
    ap_values_mm: list[float] | None = None
    ap_range_mm: tuple[float, float] | None = None
    min_area_px: float = 5.0
    simplify_px: float = 0.8
    storage_mode: Literal["disk", "memory"] = "disk"
    overwrite: bool = False

def slice_index(
        cfg: BuildConfig, 
        n_slices: int,
    ) -> list[int]:
    """
    Resolve which slice indices should be built from AP-based configuration.

    Args: 
        cfg : BuildConfig
            Configuration object containing slice selection parameters.
        n_slices : int
            Total number of slices available along the chosen orientation
            in the annotation volume. 
    Returns: 
        list[int]
            Sorted liost of unique slice indices within the valid slice range.
    Raises: 
        ValueError: if neither `ap_values_mm` nor `ap_range_mm` was provided
    """
    indices: list[int] = []

    if cfg.ap_values_mm is not None:
        indices.extend(
            ap_mm_to_slice_index(ap, resolution_um=cfg.resolution_um)
            for ap in cfg.ap_values_mm
        )

    if cfg.ap_range_mm is not None:
        ap_start, ap_end = cfg.ap_range_mm
        indices.extend(
            ap_range_mm_to_slice_indices(
                ap_start,
                ap_end,
                resolution_um=cfg.resolution_um,
            )
        )

    if not indices:
        raise ValueError(
            "No slice selection provided. Set one of: "
            "ap_values_mm or ap_range_mm."
        )

    return sorted(set(int(i) for i in indices if 0 <= int(i) < n_slices))

def download_file(
         url: str, 
         out_path: str, 
         overwrite: bool = False,
    ) -> str:
    """
    Download a file from a URL and save it to disk.

    This function streams the remote file in chunks to avoid loading
    the entire file into memory. It is intended for large assets such
    as Allen CCF annotation volumes.

    Args:
        url : str
            Remote file URL.
        out_path : str
            Destination file path on the local filesystem.
        overwrite : bool, default=False
            If False and the destination file already exists, the download
            is skipped.

    Returns:
        str
            Path to the downloaded file.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.exists(out_path) and not overwrite:
        return out_path

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return out_path

def download_bytes(
        url: str,
    ) -> bytes:
    """
    Download a remote file into memory.

    This function retrieves the entire file and returns the raw bytes
    without writing anything to disk. It is suitable for smaller files resolution
    Args: 
        url : str
            Remote file URL.
    Returns:
        bytes
            Raw file contents.
    """
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content

def load_annotation_volume(
    resolution_um: int,
    storage_mode: Literal["disk", "memory"] = "disk",
    cache_dir: str | None = None,
    overwrite: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Load the Allen CCF annotation volume.
    The annotation volume contains integer structure IDs for each voxel
    in the Allen Common Coordinate Framework (CCF). The file can either
    be cached locally on disk or downloaded directly into memory.

    Args:
        resolution_um : int
            Atlas resolution in microns (10, 25, 50, or 100).
        storage_mode : {"disk", "memory"}, default="disk"
            Loading strategy:
            * "disk"  – download the file once and cache it locally
            * "memory" – download the file into RAM without saving it
        cache_dir : str | None
            Directory used to store downloaded files when using disk mode.
        overwrite : bool, default=False
            If True, overwrite any existing cached files.
    Returns
        tuple[np.ndarray, dict]
            Annotation volume and NRRD header.
    """
    url = ANNOTATION_URLS[resolution_um]
    if storage_mode == "disk":
        if cache_dir is None:
            raise ValueError("cache_dir must be provided when storage_mode='disk'.")

        ann_path = os.path.join(cache_dir, f"annotation_{resolution_um}.nrrd")
        download_file(url=url, out_path=ann_path, overwrite=overwrite)
        volume, header = nrrd.read(ann_path)
        return volume, header

    if storage_mode == "memory":
        raw = download_bytes(url)
        volume, header = nrrd.read(io.BytesIO(raw))
        return volume, header

    raise ValueError(f"Unknown storage_mode: {storage_mode}")

def load_structure_graph(
    storage_mode: Literal["disk", "memory"] = "disk",
    cache_dir: str | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Load the Allen Brain Atlas structure ontology.
    The structure graph describes the hierarchical relationships between
    brain regions and includes region IDs, names, acronyms, and parent
    structures.

    Args: 
        storage_mode : {"disk", "memory"}, default="disk"
            Whether to cache the ontology JSON locally or load it directly
            into memory.
        cache_dir : str | None
            Directory used to store the cached JSON file in disk mode.
        overwrite : bool, default=False
            If True, overwrite existing cached files.

    Returns
        pandas.DataFrame
            Table containing structure metadata including region ID,
            name, acronym, parent structure ID, and ontology path.
    """
    if storage_mode == "disk":
        if cache_dir is None:
            raise ValueError("cache_dir must be provided when storage_mode='disk'.")

        json_path = os.path.join(cache_dir, "structure_graph.json")
        download_file(
            url=STRUCTURE_GRAPH_URL,
            out_path=json_path,
            overwrite=overwrite,
        )
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    elif storage_mode == "memory":
        raw = download_bytes(STRUCTURE_GRAPH_URL)
        data = json.loads(raw.decode("utf-8"))

    else:
        raise ValueError(f"Unknown storage_mode: {storage_mode}")

    rows = []
    def walk(node, parent_id=None):
        rows.append(
            {
                "id": int(node["id"]),
                "acronym": node.get("acronym"),
                "name": node.get("name"),
                "parent_structure_id": parent_id,
                "graph_order": node.get("graph_order"),
                "structure_id_path": node.get("structure_id_path"),
                "color_hex_triplet": node.get("color_hex_triplet"),
            }
        )
        for child in node.get("children", []):
            walk(child, parent_id=int(node["id"]))

    for root in data["msg"]:
        walk(root, parent_id=None)

    return pd.DataFrame(rows)

def get_slice_view(
        volume: np.ndarray, 
        index: int, 
        orientation: str,
    ) -> np.ndarray:
    """
    Extract a 2D slice from a 3D annotation volume.
    The Allen annotation volume stores structure IDs in a 3D grid.
    This function extracts a single slice along a specified anatomical
    orientation.

    Args:
        volume : np.ndarray
            3D annotation volume.
        index : int
            Slice index along the chosen orientation.
        orientation : {"coronal", "sagittal", "horizontal"}
            Anatomical slicing direction.
    Returns:
        np.ndarray
            2D array representing the selected slice.
    """
    if orientation == "coronal":
        return volume[:, index, :]
    if orientation == "sagittal":
        return volume[index, :, :]
    if orientation == "horizontal":
        return volume[:, :, index]
    raise ValueError(f"Unknown orientation: {orientation}")

def slice_count(
        volume: np.ndarray, 
        orientation: str,
    ) -> int:
    """
    Return the number of slices available for a given orientation.

    Args: 
        volume : np.ndarray
            3D annotation volume.
        orientation : {"coronal", "sagittal", "horizontal"}
            Anatomical slicing direction.
    Returns:
        int: number of slices along the specified axis.

    """
    if orientation == "coronal":
        return volume.shape[1]
    if orientation == "sagittal":
        return volume.shape[0]
    if orientation == "horizontal":
        return volume.shape[2]
    raise ValueError(f"Unknown orientation: {orientation}")

def mask_to_polygon(
    mask: np.ndarray,
    min_area_px: float,
    simplify_px: float,
) -> MultiPolygon | None:
    """
    Convert a binary mask into polygon geometry.
    The mask represents voxels belonging to a single brain structure
    within a slice. Contours are extracted and converted to Shapely
    polygons.

    Small disconnected fragments can optionally be removed and the
    resulting polygons simplified to reduce vertex count.

    Args:
        mask : np.ndarray
            Binary mask identifying a structure within the slice.
        min_area_px : float
            Minimum polygon area (in pixels) required to retain a connected
            component.
        simplify_px : float
            Polygon simplification tolerance in pixels.

    Returns:
        MultiPolygon | None
            Polygon geometry representing the region or None if no valid
            geometry was found.
    """
    contours = measure.find_contours(mask.astype(float), level=0.5)
    polys = []

    for contour in contours:
        xy = [(float(c[1]), float(c[0])) for c in contour]
        if len(xy) < 3:
            continue

        poly = Polygon(xy)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        if poly.area < min_area_px:
            continue

        if simplify_px > 0:
            poly = poly.simplify(simplify_px, preserve_topology=True)
            if poly.is_empty:
                continue
        polys.append(poly)
    if not polys:
        return None

    geom = unary_union(polys)
    if geom.is_empty:
        return None

    if isinstance(geom, Polygon):
        return MultiPolygon([geom])

    return geom

def build_slice_geojson(
    slice_img: np.ndarray,
    structure_df: pd.DataFrame,
    slice_index: int,
    orientation: str,
    resolution_um: int,
    min_area_px: float,
    simplify_px: float,
) -> dict:
    """
    Convert an annotation slice into a GeoJSON FeatureCollection.
    Each unique structure ID present in the slice is converted into
    polygon geometry and stored as a GeoJSON feature with associated
    metadata.

    Agrs:
        slice_img : np.ndarray
            2D annotation slice containing structure IDs.
        structure_df : pandas.DataFrame
            Table containing structure metadata from the Allen ontology.
        slice_index : int
            Index of the slice within the annotation volume.
        orientation : str
            Slice orientation ("coronal", "sagittal", or "horizontal").
        resolution_um : int
            Atlas resolution in microns.
        min_area_px : float
            Minimum polygon area required to retain a connected component.
        simplify_px : float
            Polygon simplification tolerance in pixels.

    Returns:
        dict
            GeoJSON FeatureCollection representing the slice.
    """
    unique_ids = np.unique(slice_img)
    unique_ids = unique_ids[unique_ids != 0]

    id2row = structure_df.set_index("id").to_dict(orient="index")
    features = []

    for rid in unique_ids:
        rid = int(rid)
        mask = slice_img == rid
        geom = mask_to_polygon(
            mask=mask,
            min_area_px=min_area_px,
            simplify_px=simplify_px,
        )
        if geom is None:
            continue

        row = id2row.get(rid, {})
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "Region ID": rid,
                    "Region name": row.get("name"),
                    "Acronym": row.get("acronym"),
                    "parent_structure_id": row.get("parent_structure_id"),
                    "structure_id_path": row.get("structure_id_path"),
                    "slice_index": int(slice_index),
                    "ap_mm": (
                        slice_index_to_ap_mm(
                            slice_index=slice_index,
                            resolution_um=resolution_um,
                        )
                        if orientation == "coronal"
                        else None
                    ),
                    "orientation": orientation,
                    "resolution_um": int(resolution_um),
                },
                "geometry": mapping(geom),
            }
        )

    return {"type": "FeatureCollection", "features": features}

def save_slice_geojson(
    gj: dict,
    out_path: str,
    slice_index: int,
    orientation: str,
    resolution_um: int,
) -> dict:
    """
    Save a single slice GeoJSON to disk and return its manifest metadata.

    This function writes a GeoJSON FeatureCollection representing one atlas
    slice and returns a dictionary describing the saved slice. The returned
    dictionary can later be assembled into a manifest table summarizing all
    generated slices.

    Args:
        gj : dict
            GeoJSON FeatureCollection generated by ``build_slice_geojson``.
        out_path : str
            Output path where the GeoJSON file will be written.
        slice_index : int
            Index of the slice within the annotation volume.
        orientation : str
            Slice orientation used to extract the slice
            ("coronal", "sagittal", or "horizontal").

        resolution_um : int
            Atlas voxel resolution in microns.
    Returns:
        dict
            Dictionary describing the saved slice containing:
            - ``slice_index`` : slice index in the volume
            - ``ap_mm`` : approximate AP coordinate relative to bregma
            - ``orientation`` : slice orientation
            - ``n_features`` : number of polygon regions in the slice
            - ``geojson_path`` : path to the saved GeoJSON file
    """
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gj, f)

    return {
        "slice_index": slice_index,
        "ap_mm": (
            slice_index_to_ap_mm(slice_index, resolution_um=resolution_um)
            if orientation == "coronal"
            else None
        ),
        "orientation": orientation,
        "n_features": len(gj["features"]),
        "geojson_path": out_path,
    }

def build_geojson_slices(
    volume: np.ndarray,
    structure_df: pd.DataFrame,
    slice_indices: list[int],
    out_dir: str,
    orientation: str,
    resolution_um: int,
    min_area_px: float,
    simplify_px: float,
) -> pd.DataFrame:
    """
    Build and save GeoJSON representations for multiple atlas slices.

    This function iterates over a list of slice indices, converts each slice
    of the annotation volume into polygon geometries, and writes the result
    to disk as a GeoJSON file. A manifest table summarizing all generated
    slices is returned.

    Args:
        volume : np.ndarray
            3D Allen CCF annotation volume containing structure IDs.
        structure_df : pandas.DataFrame
            Flattened Allen structure graph table containing region metadata
            (IDs, names, acronyms, parent structures).
        slice_indices : list[int]
            List of slice indices to extract and convert into GeoJSON.
        out_dir : str
            Directory where slice GeoJSON files will be saved.
        orientation : str
            Slice orientation used to extract slices from the volume
            ("coronal", "sagittal", or "horizontal").
        resolution_um : int
            Atlas voxel resolution in microns.
        min_area_px : float
            Minimum polygon area (in pixels) required for a region to be
            included. Small disconnected fragments below this threshold
            are discarded.
        simplify_px : float
            Polygon simplification tolerance in pixels. Increasing this value
            reduces vertex count and GeoJSON size.
    Returns:
        pandas.DataFrame
            Manifest table containing one row per exported slice with
            metadata including slice index, AP coordinate, orientation,
            feature count, and GeoJSON path.
    """
    os.makedirs(out_dir, exist_ok=True)
    manifest_rows = []

    for j, i in enumerate(slice_indices, start=1):
        slice_img = get_slice_view(volume, i, orientation)

        gj = build_slice_geojson(
            slice_img=slice_img,
            structure_df=structure_df,
            slice_index=i,
            orientation=orientation,
            resolution_um=resolution_um,
            min_area_px=min_area_px,
            simplify_px=simplify_px,
        )

        out_path = os.path.join(out_dir, f"slice_{i:04d}.geojson")
        row = save_slice_geojson(
            gj=gj,
            out_path=out_path,
            slice_index=i,
            orientation=orientation,
            resolution_um=resolution_um,
        )
        manifest_rows.append(row)

        print(
            f"[{j}/{len(slice_indices)}] wrote {out_path} "
            f"({len(gj['features'])} features)"
        )

    return pd.DataFrame(manifest_rows)

def build_selected_slices(cfg: BuildConfig) -> None:
    """
    Generate GeoJSON atlas slices for selected AP positions.
    This is the main pipeline function. 
    The resulting GeoJSON files can be used for visualization with Plotly
    or other geospatial visualization libraries.

    Args:
        cfg : BuildConfig
            Configuration object describing atlas resolution, slice
            selection strategy, output directory, and polygon generation
            parameters.
    Returns:
        None
            GeoJSON files and a slice manifest are written to disk.
    """
    os.makedirs(cfg.out_dir, exist_ok=True)
    geojson_dir = os.path.join(cfg.out_dir, "slices")
    os.makedirs(geojson_dir, exist_ok=True)

    ann_path = os.path.join(cfg.out_dir, f"annotation_{cfg.resolution_um}.nrrd")
    graph_path = os.path.join(cfg.out_dir, "structure_graph.json")

    download_file(
        url=ANNOTATION_URLS[cfg.resolution_um],
        out_path=ann_path,
        overwrite=cfg.overwrite,
    )
    download_file(
        url=STRUCTURE_GRAPH_URL,
        out_path=graph_path,
        overwrite=cfg.overwrite,
    )

    volume, header = nrrd.read(ann_path)
    structure_df = load_structure_graph(graph_path)

    n_slices = slice_count(volume, cfg.orientation)
    selected_indices = slice_index(cfg, n_slices=n_slices)

    print(f"Processing {len(selected_indices)} slices (atlas contains {n_slices})")

    manifest_df = build_geojson_slices(
        volume=volume,
        structure_df=structure_df,
        slice_indices=selected_indices,
        out_dir=geojson_dir,
        orientation=cfg.orientation,
        resolution_um=cfg.resolution_um,
        min_area_px=cfg.min_area_px,
        simplify_px=cfg.simplify_px,
    )

    manifest_df.to_csv(
        os.path.join(cfg.out_dir, "slice_manifest.csv"),
        index=False,
    )