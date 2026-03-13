"""
Utilities to convert between Allen CCF slice indices and
approximate AP coordinates relative to bregma.

https://community.brain-map.org/t/how-to-transform-ccf-x-y-z-coordinates-into-stereotactic-coordinates/1858
@author @anna-teruel, Jan 2026, modified by @KonradDanielewski
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CCFConfig:
    """
    Configuration for a specific Allen annotation volume resolution.

    Args:
        resolution_um : int
            Isotropic voxel size in microns.
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
    #bregma position, microns
    BREGMA_ML_UM = 5400
    BREGMA_DV_UM = 450
    BREGMA_AP_UM = 5700

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

    Args:
        ap_mm : float
            AP coordinate in mm relative to bregma.
            Example:
            -2.0 = 2.0 mm posterior to bregma
            +1.5 = 1.5 mm anterior to bregma
        resolution_um : int, default=25
            Atlas voxel resolution in microns.

    Returns
        int
            Approximate AP slice index in the annotation volume.
    """
    cfg = get_ccf_config(resolution_um)
    offset_voxels = round((ap_mm * 1000.0) / cfg.resolution_um)
    return int(cfg.bregma_ap_index + offset_voxels)


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
    offset_voxels = slice_index - cfg.bregma_ap_index
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