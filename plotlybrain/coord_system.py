"""
Utilities to convert between Allen CCF slice indices and
approximate AP coordinates relative to bregma.

https://community.brain-map.org/t/how-to-transform-ccf-x-y-z-coordinates-into-stereotactic-coordinates/1858
@author @anna-teruel, Mar 2026
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CCFConfig:
    """
    Configuration for a specific Allen annotation volume resolution.

    Args:
        resolution_um : int
            Isotropic voxel size in microns. This is directly related to which annotation volume we load 
            from the ANNOTATION_URLS in build_polygons.py. Following values are accepted: 10, 25, 50, 100. 
        bregma_ml_index : int
            Approximate mediolateral voxel index of bregma.
        bregma_dv_index : int
            Approximate dorsoventral voxel index of bregma.
        bregma_ap_index : int
            Approximate anteroposterior voxel index of bregma.
    """
    resolution_um: int
    bregma_ml_index: int
    bregma_dv_index: int
    bregma_ap_index: int

def get_ccf_config(
        resolution_um:int
) -> CCFConfig:
    """
    Compute approximate Allen CCF bregma indices for a given voxel resolution.

    The physical bregma position is assumed to be approximately: ML = 5400 µm
    DV = 450 µm, AP = 5700 µm. These are the coordinates of the approximate bregma
    location inside the Allen CCF reference space, according to the following forum
    discussion: 
    https://community.brain-map.org/t/how-to-transform-ccf-x-y-z-coordinates-into-stereotactic-coordinates/1858

    This function converts the physical position in microns into voxel indices for
    a chosen atlas resolution. 

    Args:
        resolution_um : int
            Atlas voxel size in microns. Loaded from the config class. 
            Example: 
                >>> bregma_ml_index = round(5400 / 25)  # 216
                >>> bregma_dv_index = round(450 / 25)   # 18
                >>> bregma_ap_index = round(5700 / 25)  # 228

                Meaning: 
                    In the 25 µm Allen annotation volume,
                    bregma is approximately at voxel/index:

                    ML = 216
                    DV = 18
                    AP = 228

    Returns:
        CCFConfig: Configuration object with resolution and bregma indices, based on our conversion system.
    """
    BREGMA_ML_UM = 5400 #microns
    BREGMA_DV_UM = 450 #microns
    BREGMA_AP_UM = 5700 #microns

    return CCFConfig(
        resolution_um=resolution_um,
        bregma_ml_index=round(BREGMA_ML_UM / resolution_um),
        bregma_dv_index=round(BREGMA_DV_UM / resolution_um),
        bregma_ap_index=round(BREGMA_AP_UM / resolution_um),
    )




def ap_mm_to_slice_index(
        ap_mm: float, 
        resolution_um: int = 25,
    ) -> int:
    """
    Convert AP coordinate relative to bregma (in mm) to Allen slice index.

    For a given 

    Args:
        ap_mm : float
            AP coordinate in mm relative to bregma.
        resolution_um : int, default=25 
            Atlas voxel resolution in microns. 
            This is directly related to which annotation volume we load from the 
            ANNOTATION_URLS in build_polygons.py. Following values are accepted: 10, 25, 50, 100. 

    Returns
        int:
            Approximate AP slice index in the annotation volume.
    """
    cfg = get_ccf_config(resolution_um)
    offset_voxels = round((ap_mm * 1000.0) / cfg.resolution_um)
    return int(cfg.bregma_ap_index - offset_voxels)

def slice_index_to_ap_mm(
        slice_index: int, 
        resolution_um: int = 25,
    ) -> float:
    """
    Convert Allen AP slice index to AP coordinate relative to bregma in mm.

    Args:
        slice_index : int
            Slice index in the annotation volume.
        resolution_um : int, default=25
            Atlas voxel resolution in microns.

    Returns
        float
            Approximate AP coordinate in mm relative to bregma.
    """
    cfg = get_ccf_config(resolution_um)
    offset_voxels = cfg.bregma_ap_index - slice_index
    return (offset_voxels * cfg.resolution_um) / 1000.0

def ap_range_mm_to_slice_indices(
    ap_start_mm: float,
    ap_end_mm: float,
    resolution_um: int = 25,
) -> list[int]:
    """
    Convert an AP interval in mm to a list of slice indices.

    Args: 
        ap_start_mm : float
            Start AP coordinate in mm.
        ap_end_mm : float
            End AP coordinate in mm.
        resolution_um : int, default=25
                Atlas voxel resolution in microns.

    Returns:
        list[int]
            Inclusive list of slice indices spanning the requested AP interval.
    """

    i0 = ap_mm_to_slice_index(ap_start_mm, resolution_um=resolution_um)
    i1 = ap_mm_to_slice_index(ap_end_mm, resolution_um=resolution_um)

    lo, hi = sorted((i0, i1))
    return list(range(lo, hi + 1))