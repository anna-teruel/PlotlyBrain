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
from rasterio.features import shapes
from scipy.ndimage import gaussian_filter
from skimage import measure
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.ops import unary_union

from plotlybrain.coord_system import (
    range_mm_to_slice_indices,
    slice_index_to_coordinate_mm,
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
    Configuration for building a GeoJSON from selected Allen CCF annotation slices.

    The selected slices can be defined in two ways:

        1. Explicit coordinate list:
            example: coords_mm=[2.0, -2.0, -3.9]
        2. Coordinate interval:
            example: start_mm=-3.0, end_mm=-2.0

    Args:
        out_dir : str
            Output directory where the final GeoJSON file is saved.
        resolution_um : int, default=25
            Allen annotation volume resolution in microns.
            Supported values are 10, 25, 50, and 100.
        orientation : {"coronal", "sagittal", "horizontal"}, default="coronal"
            Slice orientation used to extract 2D views from the 3D annotation
            volume. Orientation determines which stereotaxic coordinate is used:
                coronal    -> AP coordinate
                sagittal   -> ML coordinate
                horizontal -> DV coordinate
        coords_mm : list[float] | None, default=None
            Explicit list of stereotaxic coordinates in mm relative to bregma.
            Use this for one slice or selected slices.
        start_mm : float | None, default=None
            Start coordinate in mm for an interval.
        end_mm : float | None, default=None
            End coordinate in mm for an interval.
        step_mm : float | None, default=None
            Optional spacing in mm between sampled coordinates.
            If None and start_mm/end_mm are provided, all Allen slices in the
            interval are included.
        min_area_px : float, default=5.0
            Minimum polygon/component area in pixels to keep after converting
            a region mask into geometry.
        simplify_px : float, default=0.8
            Polygon simplification tolerance in pixels. Higher values reduce
            vertex count and file size, but may reduce boundary detail.
        polygon_mode : {"raster", "contour"}, default="contour"
            Method used to convert binary masks into polygons.
                "raster"  : pixel-boundary polygonization.
                "contour" : Gaussian smoothing followed by contour extraction.
        smooth_sigma : float, default=1.0
            Gaussian smoothing sigma used only when polygon_mode="contour".
        geojson_filename : str, default="atlas_slices.geojson"
            Name of the output GeoJSON file written to out_dir.
    """
    out_dir: str
    resolution_um: int = 25
    orientation: Literal["coronal", "sagittal", "horizontal"] = "coronal"

    coords_mm: list[float] | None = None
    start_mm: float | None = None
    end_mm: float | None = None
    step_mm: float | None = None

    min_area_px: float = 5.0
    simplify_px: float = 0.8
    polygon_mode: Literal["raster", "contour"] = "contour"
    smooth_sigma: float = 1.0
    geojson_filename: str = "atlas_slices.geojson"


def download_bytes(
    url: str,
) -> bytes:
    """
    Download a remote annotation volume file into memory. 
    
    It will wait max 120s (2min) for the server to respond 
    before raising an error. 

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
) -> tuple[np.ndarray, dict]:
    """
    Load an Allen CCF annotation volume into memory.

    The annotation volume contains integer structure IDs for each voxel
    in the Allen Common Coordinate Framework (CCF). The file is downloaded
    from the Allen Institute URL and read directly from memory, without
    saving the NRRD file to disk. 

    Args:
        resolution_um : int
            Atlas resolution in microns. Supported values are 10, 25, 50,
            and 100.

    Returns:
        tuple[np.ndarray, dict]
            Annotation volume and NRRD header.
    """
    if resolution_um not in ANNOTATION_URLS:
        raise ValueError(
            f"Unsupported resolution_um={resolution_um}. "
            f"Choose one of {sorted(ANNOTATION_URLS)}."
        )

    url = ANNOTATION_URLS[resolution_um]
    raw = download_bytes(url)

    memory_file = io.BytesIO(raw)
    header = nrrd.read_header(memory_file) #metadata
    volume = nrrd.read_data(header, memory_file=None)

    return volume, header

def load_structure_graph() -> pd.DataFrame:
    """
    Load the Allen Brain Atlas structure ontology into memory
    (STRUCTURE_GRAPH_URL).

    The structure graph describes the hierarchical relationships between
    brain regions and includes region IDs, names, acronyms, parent
    structures, ontology paths, and Allen display colors.

    The ontology JSON is downloaded from the Allen Institute API and read
    directly from memory, without saving the JSON file to disk.

    Returns:
        pandas.DataFrame
            Table containing structure metadata including region ID,
            acronym, name, parent structure ID, graph order, ontology path,
            and Allen color.
    """
    raw = download_bytes(STRUCTURE_GRAPH_URL)
    data = json.loads(raw.decode("utf-8"))

    rows = []
    stack = [(node, None) for node in reversed(data["msg"])] #the json is not flat

    while stack:
    # process nodes until there are no more ontology nodes left
        node, parent_id = stack.pop()
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

        children = node.get("children", [])
        for child in reversed(children):
            stack.append((child, int(node["id"])))

    return pd.DataFrame(rows)

def get_slice_view(
    volume: np.ndarray,
    index: int,
    orientation: str,
) -> np.ndarray:
    """
    Extract a 2D slice from a 3D annotation volume.

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
        return volume[index, :, :]
    if orientation == "sagittal":
        return volume[:, :, index]
    if orientation == "horizontal":
        return volume[:, index, :]
    raise ValueError(f"Unknown orientation: {orientation}")



def clean_polygons_geometry(
    polys: list[Polygon],
    min_area_px: float,
    simplify_px: float,
) -> MultiPolygon | None:
    """
    Validate, simplify, filter, and merge a list of raw Shapely polygons.
 
    Steps applied in order:
        1. Skip empty geometries.
        2. Repair invalid geometries with a zero-width buffer.
        3. Simplify with Douglas-Peucker (if simplify_px > 0).
        4. Drop components smaller than min_area_px.
        5. Merge all survivors with unary_union into a single MultiPolygon.
 
    Args:
        polys : list[Polygon]
            Raw candidate polygons from rasterization or contour extraction.
        min_area_px : float
            Minimum polygon area in pixels. Smaller components are discarded.
        simplify_px : float
            Simplification tolerance in pixels (Douglas-Peucker).
            Pass 0 to skip simplification.
 
    Returns:
        MultiPolygon | None
            Merged geometry, or None if nothing survives cleaning.
    """
    survivors: list[Polygon] = []
 
    for poly in polys:
        if poly.is_empty:
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
 
        if simplify_px > 0:
            poly = poly.simplify(simplify_px, preserve_topology=True)
            if poly.is_empty:
                continue
 
        # Flatten MultiPolygons that simplification may have produced.
        if isinstance(poly, Polygon):
            if poly.area >= min_area_px:
                survivors.append(poly)
        elif isinstance(poly, MultiPolygon):
            survivors.extend(p for p in poly.geoms if p.area >= min_area_px)
 
    if not survivors:
        return None
 
    merged = unary_union(survivors)
    if merged.is_empty:
        return None
 
    # Normalise output type to always be MultiPolygon.
    if isinstance(merged, Polygon):
        return MultiPolygon([merged])
    if isinstance(merged, MultiPolygon):
        return merged
 
    return None
 
 
def _mask_to_polygon_raster(
    mask: np.ndarray,
    min_area_px: float,
    simplify_px: float,
) -> MultiPolygon | None:
    """
    Polygonize a binary mask using rasterio pixel-boundary tracing.
 
    Produces axis-aligned polygon edges that closely follow voxel boundaries.
    Fast, but blocky at low resolutions.
 
    Args:
        mask : np.ndarray
            2D binary mask (will be cast to uint8 internally).
        min_area_px : float
            Minimum polygon area in pixels.
        simplify_px : float
            Simplification tolerance in pixels.
 
    Returns:
        MultiPolygon | None
    """
    mask_u8 = mask.astype(np.uint8)
    if mask_u8.max() == 0:
        return None
 
    polys: list[Polygon] = []
    for geom_dict, value in shapes(mask_u8, mask=mask_u8 > 0):
        if value != 1:
            continue
        geom = shape(geom_dict)
        if geom.is_empty:
            continue
        if isinstance(geom, Polygon):
            polys.append(geom)
        elif isinstance(geom, MultiPolygon):
            polys.extend(geom.geoms)
 
    return clean_polygons_geometry(polys, min_area_px, simplify_px)
 
 
def _mask_to_polygon_contour(
    mask: np.ndarray,
    min_area_px: float,
    simplify_px: float,
    smooth_sigma: float,
) -> MultiPolygon | None:
    """
    Polygonize a binary mask via Gaussian smoothing + iso-contour extraction.
 
    Produces smoother boundaries than raster mode, especially at coarse
    resolutions. Slower due to the smoothing step.
 
    Args:
        mask : np.ndarray
            2D binary mask.
        min_area_px : float
            Minimum polygon area in pixels.
        simplify_px : float
            Simplification tolerance in pixels.
        smooth_sigma : float
            Gaussian smoothing sigma applied before contour extraction.
 
    Returns:
        MultiPolygon | None
    """
    mask_f = mask.astype(float)
    if mask_f.max() == 0:
        return None
 
    smoothed = gaussian_filter(mask_f, sigma=smooth_sigma)
    contours = measure.find_contours(smoothed, level=0.5)
 
    polys: list[Polygon] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        # skimage returns (row, col); Shapely expects (x, y) = (col, row).
        coords = [(float(c[1]), float(c[0])) for c in contour]
        poly = Polygon(coords)
        if not poly.is_empty:
            polys.append(poly)
 
    return clean_polygons_geometry(polys, min_area_px, simplify_px)
 
 
def mask_to_polygon(
    mask: np.ndarray,
    min_area_px: float,
    simplify_px: float,
    polygon_mode: Literal["raster", "contour"] = "contour",
    smooth_sigma: float = 1.0,
) -> MultiPolygon | None:
    """
    Convert a binary mask into cleaned polygon geometry.
 
    Dispatches to raster or contour extraction based on polygon_mode, then
    passes the result through clean_polygons_geometry.
 
    Args:
        mask : np.ndarray
            2D binary mask for a single brain region.
        min_area_px : float
            Minimum polygon area in pixels.
        simplify_px : float
            Simplification tolerance in pixels.
        polygon_mode : {"raster", "contour"}, default="contour"
            "raster" is faster; "contour" produces smoother boundaries.
        smooth_sigma : float, default=1.0
            Gaussian sigma, used only when polygon_mode="contour".
 
    Returns:
        MultiPolygon | None
            Cleaned geometry, or None if the mask is empty or too small.
    """
    if polygon_mode == "raster":
        return _mask_to_polygon_raster(mask, min_area_px, simplify_px)
 
    if polygon_mode == "contour":
        return _mask_to_polygon_contour(mask, min_area_px, simplify_px, smooth_sigma)
 
    raise ValueError(f"Unknown polygon_mode: {polygon_mode!r}. Choose 'raster' or 'contour'.")
 
 
# ---------------------------------------------------------------------------
# GeoJSON construction
# ---------------------------------------------------------------------------
 
def build_slice_features(
    slice_img: np.ndarray,
    structure_df: pd.DataFrame,
    slice_index: int,
    orientation: str,
    resolution_um: int,
    min_area_px: float,
    simplify_px: float,
    polygon_mode: Literal["raster", "contour"] = "contour",
    smooth_sigma: float = 1.0,
) -> list[dict]:
    """
    Convert one 2D annotation slice into a list of GeoJSON feature dicts.
 
    Each brain region present in the slice becomes one feature. Every feature
    embeds its slice_index and coordinate_mm so the unified GeoJSON can be
    filtered by slice without auxiliary files.
 
    Args:
        slice_img : np.ndarray
            2D array of Allen structure IDs for this slice.
        structure_df : pd.DataFrame
            Allen ontology table from load_structure_graph().
        slice_index : int
            Index of this slice in the annotation volume.
        orientation : str
            Slice orientation.
        resolution_um : int
            Atlas voxel resolution in microns.
        min_area_px : float
            Minimum polygon area in pixels.
        simplify_px : float
            Simplification tolerance in pixels.
        polygon_mode : {"raster", "contour"}, default="contour"
            Polygon extraction method.
        smooth_sigma : float, default=1.0
            Gaussian sigma for contour mode.
 
    Returns:
        list[dict]
            GeoJSON feature dicts for every region that produced valid geometry.
    """
    unique_ids = np.unique(slice_img)
    unique_ids = unique_ids[unique_ids != 0]
 
    id2row = structure_df.set_index("id").to_dict(orient="index")
    coordinate_mm = slice_index_to_coordinate_mm(
        slice_index=slice_index,
        orientation=orientation,
        resolution_um=resolution_um,
    )
 
    features: list[dict] = []
    for rid in unique_ids:
        rid = int(rid)
        geom = mask_to_polygon(
            mask=(slice_img == rid),
            min_area_px=min_area_px,
            simplify_px=simplify_px,
            polygon_mode=polygon_mode,
            smooth_sigma=smooth_sigma,
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
                    "coordinate_mm": coordinate_mm,
                    "orientation": orientation,
                    "resolution_um": int(resolution_um),
                },
                "geometry": mapping(geom),
            }
        )
 
    return features
 
 
def build_unified_geojson(
    volume: np.ndarray,
    structure_df: pd.DataFrame,
    slice_indices: list[int],
    orientation: str,
    resolution_um: int,
    min_area_px: float,
    simplify_px: float,
    polygon_mode: Literal["raster", "contour"] = "contour",
    smooth_sigma: float = 1.0,
) -> dict:
    """
    Build a single GeoJSON FeatureCollection for one or more atlas slices.
 
    Iterates over every requested slice, converts each region mask into
    polygon geometry, and accumulates all features into one FeatureCollection.
    Works identically for num_slices=1 and num_slices>1.
 
    Args:
        volume : np.ndarray
            3D Allen CCF annotation volume.
        structure_df : pd.DataFrame
            Allen ontology table.
        slice_indices : list[int]
            Slice indices to include (may be length 1).
        orientation : str
            Slice orientation.
        resolution_um : int
            Atlas voxel resolution in microns.
        min_area_px : float
            Minimum polygon area in pixels.
        simplify_px : float
            Simplification tolerance in pixels.
        polygon_mode : {"raster", "contour"}, default="contour"
            Polygon extraction method.
        smooth_sigma : float, default=1.0
            Gaussian sigma for contour mode.
 
    Returns:
        dict
            GeoJSON FeatureCollection containing all features from all slices.
    """
    all_features: list[dict] = []
    n = len(slice_indices)
 
    for j, i in enumerate(slice_indices, start=1):
        features = build_slice_features(
            slice_img=get_slice_view(volume, i, orientation),
            structure_df=structure_df,
            slice_index=i,
            orientation=orientation,
            resolution_um=resolution_um,
            min_area_px=min_area_px,
            simplify_px=simplify_px,
            polygon_mode=polygon_mode,
            smooth_sigma=smooth_sigma,
        )
        all_features.extend(features)
        print(f"[{j}/{n}] slice {i:04d} → {len(features)} features (total: {len(all_features)})")
 
    return {"type": "FeatureCollection", "features": all_features}
 
 
def save_geojson(geojson_obj: dict, out_path: str) -> str:
    """
    Save a GeoJSON FeatureCollection to disk.
 
    Args:
        geojson_obj : dict
            GeoJSON FeatureCollection to serialise.
        out_path : str
            Destination file path. Parent directories are created if needed.
 
    Returns:
        str
            Absolute path to the written file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson_obj, f)
    return os.path.abspath(out_path)
 
 
# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------
 
def build_selected_slices(cfg: BuildConfig) -> str:
    """
    Build a unified GeoJSON FeatureCollection for the requested atlas slices.
 
    Downloads the annotation volume and structure ontology, resolves slice
    indices from the coordinate configuration, builds polygon geometry for
    every region in every slice, and writes a single GeoJSON file to disk.
 
    The output file can be loaded at any later time and mapped with a score
    table before rendering with Plotly — no re-processing of the volume needed.
 
    Args:
        cfg : BuildConfig
            Full build configuration.
 
    Returns:
        str
            Absolute path to the saved GeoJSON file.
    """
    os.makedirs(cfg.out_dir, exist_ok=True)
 
    print(f"Loading annotation volume at {cfg.resolution_um} µm resolution...")
    volume, _ = load_annotation_volume(cfg.resolution_um)
 
    print("Loading structure ontology...")
    structure_df = load_structure_graph()
 
    n_slices = slice_count(volume, cfg.orientation)
    selected_indices = resolve_slice_indices(cfg, n_slices=n_slices)
 
    if not selected_indices:
        raise ValueError(
            "No valid slice indices resolved from the provided coordinates. "
            "Check that your coordinate values fall within the atlas bounds."
        )
 
    print(
        f"Processing {len(selected_indices)} slice(s) "
        f"(atlas contains {n_slices} along {cfg.orientation} axis)"
    )
 
    geojson_obj = build_unified_geojson(
        volume=volume,
        structure_df=structure_df,
        slice_indices=selected_indices,
        orientation=cfg.orientation,
        resolution_um=cfg.resolution_um,
        min_area_px=cfg.min_area_px,
        simplify_px=cfg.simplify_px,
        polygon_mode=cfg.polygon_mode,
        smooth_sigma=cfg.smooth_sigma,
    )
 
    out_path = save_geojson(
        geojson_obj=geojson_obj,
        out_path=os.path.join(cfg.out_dir, cfg.geojson_filename),
    )
 
    print(
        f"Saved → {out_path} "
        f"({len(selected_indices)} slice(s), {len(geojson_obj['features'])} features)"
    )
    return out_path
 
