"""
Region-level score tables from QUINT *_RefAtlasRegions.csv exports.
@author Anna Teruel-Sanchis, Jan 2026
"""

import glob
import os
from typing import Callable, Literal

import pandas as pd
from scipy.stats import zscore

ScoreName = Literal["rel_abundance", "frequency"]
RelAbundanceMethod = Literal["within", "reference"]
ScoreFn = Callable[[pd.DataFrame], pd.DataFrame]


def find_animal_id(filename: str) -> str:
	"""
	Extract the animal identifier from a QUINT region CSV filename.

	Args:
	    filename: Basename or full path to a QUINT
	                   *_RefAtlasRegions.csv file.

	Returns:
	    Collapsed animal ID string.
	"""
	base = os.path.basename(filename)
	base = base.split("_RefAtlasRegions")[0]

	return base.split("-")[0]

def load_refatlas_regions(
	data_dir: str,
	sep: str = ";",
	col_id: str = "Region ID",
	col_name: str = "Region name",
	col_count: str = "Object count",
	exclude_region_ids: set[int] = {0, 997},
) -> pd.DataFrame:
	"""
	Load and concatenate QUINT *_RefAtlasRegions.csv files from a directory.

	Args:
	    data_dir: Folder containing QUINT *_RefAtlasRegions.csv exports.
	    sep: CSV separator used by QUINT exports (often ';').
	    col_id: Column name for the Allen region ID.
	    col_name: Column name for the region name.
	    col_count: Column name for the object count.
	    exclude_region_ids: Region IDs to exclude
	                    (e.g. background/root).

	Returns:
	    Long-format table with columns ['animal', col_id, col_name, col_count].

	Raises:
	    FileNotFoundError: If no matching files exist in data_dir.
	"""
	files = sorted(glob.glob(os.path.join(data_dir, "*_RefAtlasRegions.csv")))
	if not files:
		raise FileNotFoundError(f"No *_RefAtlasRegions.csv files found in: {data_dir}")

	out = []
	for file in files:
		df = pd.read_csv(file, sep=sep, engine="python")[[col_id, col_name, col_count]].copy()
		df["animal"] = find_animal_id(file)
		df[col_id] = pd.to_numeric(df[col_id], errors="coerce").astype("Int64")
		df[col_count] = pd.to_numeric(df[col_count], errors="coerce").fillna(0)
		if exclude_region_ids:
			df = df[~df[col_id].isin(exclude_region_ids)]
		out.append(df)
	return pd.concat(out, ignore_index=True)

def compute_animal_region_counts(
	df: pd.DataFrame,
	col_id: str = "Region ID",
	col_name: str = "Region name",
	col_count: str = "Object count",
) -> pd.DataFrame:
	"""
	Collapse raw QUINT rows into per-animal, per-region object counts.

	In QUINT exports, the same brain region may appear multiple times for a
	given animal. For example, when a region is subdivided into anatomical
	subregions such as parts of CA1 (e.g. cornus amonis). This function aggregates 
	those rows so that each animal–region pair has a single value representing the 
	total number of detected objects. For downstream analysis, having one value per 
	animal and region is required.

	Args:
	    df: Output of load_refatlas_regions().
	    col_id: Region ID column.
	    col_name: Region name column.
	    col_count: Object count column.

	Returns:
	    DataFrame with columns: [animal, col_id, col_name, objects]
	    where 'objects' is the summed count per (animal, region).
	"""
	return (
		df.groupby(["animal", col_id, col_name], dropna=True)
		.agg(objects=(col_count, "sum"))
		.reset_index()
	)

def compute_region_counts(
    region_by_subject: pd.DataFrame,
    col_id: str = "Region ID",
    col_name: str = "Region name",
) -> pd.DataFrame:
    """
    Aggregate per-animal counts into one row per region.

	This function summarizes the per-animal region table by collapsing
	across animals. For each brain region, it computes the total number of
	detected objects across all animals and the number of animals that
	contributed observations to that region.

    Returns:
        DataFrame with columns:
            [col_id, col_name, objects_total, n_animals]
    """
    return (
        region_by_subject.groupby([col_id, col_name], dropna=True)
        .agg(
            objects_total=("objects", "sum"),
            n_animals=("animal", "nunique"),
        )
        .reset_index()
    )


def compute_reference_stats(
    region_by_subject: pd.DataFrame,
    col_id: str = "Region ID",
    col_name: str = "Region name",
) -> dict[str, float]:
    """
    Fit a shared reference distribution for relative abundance z-scoring.

    The reference is computed from region-level total object counts
    aggregated across the supplied reference dataset.

    Args:
        region_by_subject: Output of compute_animal_region_counts() for the reference dataset.
        col_id: Region ID column.
        col_name: Region name column.

    Returns:
        dict with:
            - reference_mean
            - reference_std
            - n_regions_reference
    """
    region_totals = compute_region_counts(
        region_by_subject,
        col_id=col_id,
        col_name=col_name,
    )

    ref_mean = float(region_totals["objects_total"].mean())
    ref_std = float(region_totals["objects_total"].std(ddof=0))

    if ref_std == 0:
        raise ValueError("Reference standard deviation is 0; cannot compute z-scores.")

    return {
        "reference_mean": ref_mean,
        "reference_std": ref_std,
        "n_regions_reference": int(len(region_totals)),
    }


def relative_abundance(
    region_by_subject: pd.DataFrame,
    col_id: str = "Region ID",
    col_name: str = "Region name",
    method: RelAbundanceMethod = "within",
    reference_stats: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Compute region-level relative abundance of objects across animals.

    Relative abundance quantifies whether a brain region contains more or fewer
	detected objects than expected relative to the overall distribution of objects
	across the atlas. The metric is based on the total number of objects detected
	in each region across all animals and expresses this value as a z-score.

	Two modes are supported:
    - method="within":
        z-score region totals within the same dataset
        (old behavior; not cross-cohort comparable). 
		This parameter is only meaningful inside a given cohort. 
    - method="reference":
        z-score region totals using a shared reference mean/std
        computed from an external pooled or control dataset
        (cross-cohort comparable)

    Args:
        region_by_subject: Output of compute_animal_region_counts(), containing
                           per-animal object counts for each region.
        col_id: Column name for the Allen region ID.
        col_name: Column name for the region name.
        method: "within" or "reference".
        reference_stats: Output of compute_reference_stats()
                         when method="reference".

    Returns:
        Region-level table with columns:
            - col_id
            - col_name
            - objects_total
            - n_animals
            - relative_abundance_z
            - rel_abundance_method
            - reference_mean (if method="reference")
            - reference_std (if method="reference")
    """
    region_abundance = compute_region_counts(
        region_by_subject,
        col_id=col_id,
        col_name=col_name,
    )

    if method == "within":
        region_abundance["relative_abundance_z"] = zscore(
            region_abundance["objects_total"],
            ddof=0,
            nan_policy="omit",
        )
        region_abundance["rel_abundance_method"] = "within"

    elif method == "reference":
        if reference_stats is None:
            raise ValueError(
                "reference_stats must be provided when method='reference'."
            )

        ref_mean = float(reference_stats["reference_mean"])
        ref_std = float(reference_stats["reference_std"])

        if ref_std == 0:
            raise ValueError("reference_std is 0; cannot compute z-scores.")

        region_abundance["relative_abundance_z"] = (
            region_abundance["objects_total"] - ref_mean
        ) / ref_std
        region_abundance["rel_abundance_method"] = "reference"
        region_abundance["reference_mean"] = ref_mean
        region_abundance["reference_std"] = ref_std

    else:
        raise ValueError("method must be one of {'within', 'reference'}.")

    return region_abundance

def consistency_score(
	region_by_subject: pd.DataFrame,
	col_id: str = "Region ID",
	col_name: str = "Region name",
) -> pd.DataFrame:
	"""
	Compute region-level frequency (consistency) of object presence across animals.
	Frequency is defined as the fraction of animals with at least one object in a
	given brain region. This provides a measure of how consistently a signal is
	observed across the cohort, independent of the absolute object counts.

	Args:
	    region_by_subject: Output of compute_animal_region_counts, containing
	                    one row per (animal, region) with a per-animal object count
	                    stored in the 'objects' column.
	    col_id: Column name for the Allen region ID.
	    col_name: Column name for the region name.

	Returns:
	    pandas.DataFrame: Region-level table with columns:
	        - col_id: Allen region ID
	        - col_name: Region name
	        - n_true_animals: Number of animals with objects > 0
	        - total_animals: Total number of animals in the dataset
	        - frequency: Fraction of animals with objects > 0
	"""
	total_animals = int(region_by_subject["animal"].nunique())

	region_freq = (
		region_by_subject.assign(has_objects=lambda d: d["objects"] > 0)
		.groupby([col_id, col_name], dropna=True)
		.agg(
			n_true_animals=("has_objects", "sum"),
		)
		.reset_index()
	)

	region_freq["n_true_animals"] = region_freq["n_true_animals"].astype(int)
	region_freq["total_animals"] = total_animals
	region_freq["frequency"] = region_freq["n_true_animals"] / total_animals

	region_freq = region_freq.sort_values(
		["frequency", "n_true_animals", col_name],
		ascending=[False, False, True],
	).reset_index(drop=True)

	return region_freq


def save_scores(
	data_dir: str,
	out_path: str,
	score: ScoreName = "rel_abundance",
	score_fn: ScoreFn | None = None,
	sep: str = ";",
	col_id: str = "Region ID",
	col_name: str = "Region name",
	col_count: str = "Object count",
	exclude_region_ids: set[int] = {0, 997},
) -> pd.DataFrame:
	"""
	Compute a region-level score table from QUINT exports and save it to disk.

	Args:
	    data_dir: Folder containing QUINT *_RefAtlasRegions.csv exports.
	    out_path: Output CSV path.
	    score: Which score to compute. 2 options:
	                    - "rel_abundance" (z-scored total object counts)
	                    - "frequency" (fraction of animals with objects > 0)
	    score_fn: Custom scoring function. If provided, it
	                    overrides `score` and must accept the collapsed
	                    region_by_subject table and return a region-level table.
	    sep: CSV separator used by QUINT exports (often ';').
	    col_id: Column name for the Allen region ID.
	    col_name: Column name for the region name.
	    col_count: Column name for the object count.
	    exclude_region_ids: Region IDs to exclude (e.g. background/root).

	Returns:
	    Saved score table.

	Raises:
	    ValueError: If `score` is not recognized.
	"""
	df = load_refatlas_regions(
		data_dir,
		sep=sep,
		col_id=col_id,
		col_name=col_name,
		col_count=col_count,
		exclude_region_ids=exclude_region_ids,
	)
	region_by_subject = compute_animal_region_counts(
		df,
		col_id=col_id,
		col_name=col_name,
		col_count=col_count,
	)

	if score_fn is None:
		scorers: dict[str, ScoreFn] = {
			"rel_abundance": lambda df: relative_abundance(df, col_id=col_id, col_name=col_name),
			"frequency": lambda df: consistency_score(df, col_id=col_id, col_name=col_name),
		}
		try:
			score_fn = scorers[score]
		except KeyError as e:
			raise ValueError(f"Unknown score='{score}'. Use one of {list(scorers.keys())}.") from e

	score_df = score_fn(region_by_subject)
	os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
	score_df.to_csv(out_path, index=False)
	return score_df
