"""Tests for MetadataConfig: loading, group-column normalization, merging."""

import pandas as pd
import pytest

from plotlybrain.metadata import MetadataConfig


def test_load_none_path_returns_none():
	assert MetadataConfig(metadata_path=None).load() is None


def test_load_strips_column_whitespace(metadata_csv):
	cfg = MetadataConfig(metadata_path=metadata_csv, sep=",", animal_col="animal")
	meta = cfg.load()
	assert "group" in meta.columns  # " group " -> "group"
	assert "sex" in meta.columns


def test_load_missing_animal_col_raises(metadata_csv):
	cfg = MetadataConfig(metadata_path=metadata_csv, sep=",", animal_col="subject")
	with pytest.raises(KeyError):
		cfg.load()


@pytest.mark.parametrize(
	"group_col, expected",
	[
		(None, None),
		("group", ["group"]),
		(["group", "sex"], ["group", "sex"]),
	],
)
def test_group_cols_normalization(group_col, expected):
	assert MetadataConfig(group_col=group_col).group_cols() == expected


def test_merge_and_add_groups_single_column(metadata_csv):
	cfg = MetadataConfig(
		metadata_path=metadata_csv, sep=",", animal_col="animal", group_col="group"
	)
	df = pd.DataFrame({"animal": ["A1", "A3"], "objects": [10, 5]})
	merged, group_cols = cfg.merge_and_add_groups(df)
	assert group_cols == ["group"]
	assert merged.loc[merged["animal"] == "A1", "group_label"].iloc[0] == "control"
	assert merged.loc[merged["animal"] == "A3", "group_label"].iloc[0] == "treated"


def test_merge_and_add_groups_multi_column_label(metadata_csv):
	cfg = MetadataConfig(
		metadata_path=metadata_csv,
		sep=",",
		animal_col="animal",
		group_col=["group", "sex"],
		group_name_sep="_",
	)
	df = pd.DataFrame({"animal": ["A1"], "objects": [10]})
	merged, group_cols = cfg.merge_and_add_groups(df)
	assert group_cols == ["group", "sex"]
	assert merged["group_label"].iloc[0] == "control_M"


def test_merge_and_add_groups_no_metadata_no_groups():
	cfg = MetadataConfig(metadata_path=None, group_col=None)
	df = pd.DataFrame({"animal": ["A1"], "objects": [10]})
	merged, group_cols = cfg.merge_and_add_groups(df)
	assert group_cols is None
	assert "group_label" not in merged.columns


def test_merge_and_add_groups_missing_group_column_raises():
	# No metadata file, but a grouping column is requested that does not exist.
	cfg = MetadataConfig(metadata_path=None, group_col="nope")
	df = pd.DataFrame({"animal": ["A1"], "objects": [10]})
	with pytest.raises(KeyError):
		cfg.merge_and_add_groups(df)
