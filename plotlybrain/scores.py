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


def collapse_animal(
	df: pd.DataFrame,
	col_id: str = "Region ID",
	col_name: str = "Region name",
	col_count: str = "Object count",
) -> pd.DataFrame:
	"""
	Collapse raw rows into per-animal, per-region total object counts.
	After loading the data, it might contain multiple rows per animal and region.
	For downstream analysis, you want one value per animal.

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


def relative_abundance(
	region_by_subject: pd.DataFrame,
	col_id: str = "Region ID",
	col_name: str = "Region name",
) -> pd.DataFrame:
	"""
	Compute region-level relative abundance of objects across animals.

	For each brain region, object counts are summed across all animals
	to obtain a total abundance measure. These totals are then
	z-scored across regions to quantify relative enrichment or
	depletion with respect to the atlas-wide mean.

	Args:
	    region_by_subject: output of
	                    collapse_animal, containing
	                    per-animal object counts for each region.
	    col_id: column name for the Allen region ID.
	    col_name: column name for the region name.

	Returns:
	    pandas.DataFrame: Region-level table with columns:
	        - col_id: Allen region ID
	        - col_name: Region name
	        - objects_total: Total number of objects across animals
	        - n_animals: Number of animals contributing to the region
	        - relative_abundance_z: Z-scored relative abundance
	"""
	region_abundance = (
		region_by_subject.groupby([col_id, col_name], dropna=True)
		.agg(
			objects_total=("objects", "sum"),
			n_animals=("animal", "nunique"),
		)
		.reset_index()
	)

	region_abundance["relative_abundance_z"] = zscore(
		region_abundance["objects_total"],
		ddof=0,  # delta degrees of freedom, 0 means population std
		nan_policy="omit",
	)
	return region_abundance


def frequency_score(
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
	    region_by_subject: Output of collapse_animal, containing
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
	region_by_subject = collapse_animal(
		df,
		col_id=col_id,
		col_name=col_name,
		col_count=col_count,
	)

	if score_fn is None:
		scorers: dict[str, ScoreFn] = {
			"rel_abundance": lambda df: relative_abundance(df, col_id=col_id, col_name=col_name),
			"frequency": lambda df: frequency_score(df, col_id=col_id, col_name=col_name),
		}
		try:
			score_fn = scorers[score]
		except KeyError as e:
			raise ValueError(f"Unknown score='{score}'. Use one of {list(scorers.keys())}.") from e

	score_df = score_fn(region_by_subject)
	os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
	score_df.to_csv(out_path, index=False)
	return score_df
