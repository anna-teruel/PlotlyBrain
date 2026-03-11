"""
Region-level score tables from QUINT *_RefAtlasRegions.csv exports.
@author Anna Teruel-Sanchis, Jan 2026
"""

import glob
import os
from typing import Callable, Literal

import pandas as pd
from scipy.stats import zscore

ScoreName = Literal["rel_abundance", "frequency", "density"]
RelAbundanceMethod = Literal["within", "reference"]
ReferenceMode = Literal["pooled", "group"]
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
	col_area: str = "Region area",
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
		col_area: Column name for the region area (typically in squared pixels).
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
		df = pd.read_csv(file, sep=sep, engine="python")[[col_id, col_name, col_count, col_area]].copy()
		df["animal"] = find_animal_id(file)
		df[col_id] = pd.to_numeric(df[col_id], errors="coerce").astype("Int64")
		df[col_count] = pd.to_numeric(df[col_count], errors="coerce").fillna(0)
		df[col_area] = pd.to_numeric(df[col_area], errors="coerce").fillna(0)

		if exclude_region_ids:
			df = df[~df[col_id].isin(exclude_region_ids)]
		out.append(df)
	return pd.concat(out, ignore_index=True)

def compute_animal_region_counts(
	df: pd.DataFrame,
	col_id: str = "Region ID",
	col_name: str = "Region name",
	col_count: str = "Object count",
	col_area: str = "Region area",
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
		.agg(
			objects=(col_count, "sum"),
			region_area=(col_area, "mean"),
			)
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

	This score is cohort-dependent. Should be run for each cohort independently. 

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

def density_score(
    region_by_subject: pd.DataFrame,
    col_id: str = "Region ID",
    col_name: str = "Region name",
    area_col: str = "region_area",
) -> pd.DataFrame:
    """
    Compute region-level object density across animals.

    Density is defined as the total number of detected objects divided by the
    total region area across animals.

    Interpretation:
        - Higher density values indicate that objects are more concentrated
          within that region.
        - Lower density values indicate sparser signal.

    Args:
        region_by_subject: DataFrame with one row per (animal, region),
            containing columns 'objects' and region area.
        col_id: Column name for the Allen region ID.
        col_name: Column name for the region name.
        area_col: Column containing per-animal region area.

    Returns:
        pandas.DataFrame with one row per region and columns:
            - col_id
            - col_name
            - objects_total
            - area_total
            - n_animals
            - density
    """
    region_density = (
        region_by_subject.groupby([col_id, col_name], dropna=True)
        .agg(
            objects_total=("objects", "sum"),
            area_total=(area_col, "mean"),
            n_animals=("animal", "nunique"),
        )
        .reset_index()
    )

    region_density["density"] = (
        region_density["objects_total"] / region_density["area_total"]
    )
    region_density.loc[region_density["area_total"] <= 0, "density"] = pd.NA

    return region_density

def save_scores(
    data_dir: str,
    out_path: str,
    score: ScoreName = "rel_abundance",
    score_fn: ScoreFn | None = None,
    sep: str = ";",
    col_id: str = "Region ID",
    col_name: str = "Region name",
    col_count: str = "Object count",
    col_area: str = "Region area",
    exclude_region_ids: set[int] = {0, 997},
    metadata_path: str | None = None,
    metadata_sep: str | None = None,
    animal_col: str = "animal",
    group_col: str | list[str] | None = None,
    group_name_sep: str = "_",
    rel_abundance_method: RelAbundanceMethod = "within",
    reference_mode: ReferenceMode = "pooled",
    reference_group: str | list[str] | None = None,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """
    Compute region-level score tables from QUINT exports and save them to disk.

    This function can operate on the full dataset or, if metadata and a
    grouping column are provided, compute one score table per group.

    For `score="rel_abundance"`, two normalization modes are supported:
        - "within": z-score region totals within each dataset/group
        - "reference": z-score region totals using shared reference stats

    If grouping is used together with `rel_abundance_method="reference"`,
    reference statistics can be computed from:
        - all animals pooled together (`reference_mode="pooled"`)
        - one reference group (`reference_mode="group"`)

    Args:
        data_dir: Folder containing QUINT *_RefAtlasRegions.csv exports.
        out_path: Output CSV path. If `group_col` is provided, this is treated
            as an output prefix and one file per group is written.
        score: Which score to compute:
            - "rel_abundance": z-scored regional abundance
            - "frequency": fraction of animals with objects > 0
            - "density": objects per unit area
        score_fn: Optional custom scoring function. If provided, it overrides
            `score` and must accept the per-animal region table.
        sep: CSV separator used by QUINT exports.
        col_id: Column name for the Allen region ID.
        col_name: Column name for the region name.
        col_count: Column name for the object count.
        col_area: Column name for the region area in the raw QUINT table.
        exclude_region_ids: Region IDs to exclude (e.g. background/root).
        metadata_path: Optional metadata CSV containing at least one row per animal.
        metadata_sep: Separator used in the metadata file.
        animal_col: Column used to merge metadata with QUINT data.
        group_col: Metadata column(s) used to define groups. Can be a single
            column name or a list of column names.
        group_name_sep: how you want to separate different group categories. Default "_"
        rel_abundance_method: "within" or "reference" for relative abundance.
        reference_mode: How to compute shared reference stats when
            `rel_abundance_method="reference"`:
            - "pooled": use all animals
            - "group": use only `reference_group`
        reference_group: Group name to use when `reference_mode="group"`.

    Returns:
        If `group_col` is None:
            A single score DataFrame.
        If `group_col` is provided:
            A dictionary mapping each group name to its score DataFrame.

    Raises:
        ValueError: If `score` is invalid or reference settings are inconsistent.
    """
    df = load_refatlas_regions(
        data_dir=data_dir,
        sep=sep,
        col_id=col_id,
        col_name=col_name,
        col_count=col_count,
        col_area=col_area,
        exclude_region_ids=exclude_region_ids,
    )
    
    region_by_subject = compute_animal_region_counts(
         df,
         col_id = col_id, 
         col_name = col_name, 
         col_count = col_count, 
         col_area = col_area,
	)
    
    #adding group information from metadata
    if metadata_path is not None:
        meta = pd.read_csv(metadata_path, sep=metadata_sep, engine="python")
        meta.columns = meta.columns.str.strip()

        if animal_col not in meta.columns:
            raise KeyError(
                f"Column '{animal_col}' not found in metadata file. "
                f"Available columns: {list(meta.columns)}"
            )

        region_by_subject = region_by_subject.merge(meta, on=animal_col, how="left")
    
	#combining group labels into one category (e.g sex + genotype --> WT_F)
    if isinstance(group_col, str):
        group_cols = [group_col]
    else:
         group_cols = group_col

    if group_cols is not None:
        missing = [c for c in group_cols if c not in region_by_subject.columns]
        if missing:
            raise KeyError(
                f"Grouping column(s) not found after loading/merging data: {missing}. "
                f"Available columns: {list(region_by_subject.columns)}"
            )

        region_by_subject["group_label"] = (
            region_by_subject[group_cols]
            .astype(str)
            .agg(group_name_sep.join, axis=1)
        )
          
	# for rel_abundance score:
    reference_stats = None
    if score == "rel_abundance" and rel_abundance_method == "reference":
        if group_col is None or reference_mode == "pooled":
            ref_df = region_by_subject

        elif reference_mode == "group":
            if reference_group is None:
                raise ValueError(
                    "reference_group must be provided when reference_mode='group'."
                )
            ref_df = region_by_subject[region_by_subject[group_col] == reference_group]

            if ref_df.empty:
                raise ValueError(
                    f"No rows found for reference_group='{reference_group}'."
                )
        else:
            raise ValueError("reference_mode must be one of {'pooled', 'group'}.")

        reference_stats = compute_reference_stats(
            ref_df,
            col_id=col_id,
            col_name=col_name,
        )

    # build scorer
    if score_fn is None:
        scorers: dict[str, ScoreFn] = {
            "rel_abundance": lambda d: relative_abundance(
                d,
                col_id=col_id,
                col_name=col_name,
                method=rel_abundance_method,
                reference_stats=reference_stats,
            ),
            "frequency": lambda d: consistency_score(
                d,
                col_id=col_id,
                col_name=col_name,
            ),
            "density": lambda d: density_score(
                d,
                col_id=col_id,
                col_name=col_name,
                area_col="region_area",
            ),
        }

        try:
            score_fn = scorers[score]
        except KeyError as e:
            raise ValueError(
                f"Unknown score='{score}'. Use one of {list(scorers.keys())}."
            ) from e

	# save df
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if group_col is None:
        score_df = score_fn(region_by_subject)
        score_df.to_csv(out_path, index=False)
        return score_df

    # one file per group
    out_dir = os.path.dirname(out_path)
    base_name = os.path.splitext(os.path.basename(out_path))[0]
    results: dict[str, pd.DataFrame] = {}

    for group, sub_df in region_by_subject.groupby('group_label', dropna=False):
        group_str = str(group)
        group_score_df = score_fn(sub_df)

        group_out = os.path.join(out_dir, f"{score}_{group_str}.csv")
        group_score_df.to_csv(group_out, index=False)
        results[group_str] = group_score_df

    return results