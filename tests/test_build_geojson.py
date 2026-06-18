"""Tests for slice extraction, mask polygonization, and GeoJSON assembly."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, Polygon

import plotlybrain.build_geoJSON as bg
from plotlybrain.build_geoJSON import (
	load_annotation_volume,
	load_structure_graph,
	get_slice_view,
	clean_polygons_geometry,
	mask_to_polygon,
	build_geojson,
	save_geojson,
)


def _flat_coords(geometry_coordinates):
	return np.array(
		[pt for poly in geometry_coordinates for ring in poly for pt in ring]
	)


def test_load_annotation_volume_rejects_unsupported_resolution():
	# Validation happens before any network access.
	with pytest.raises(ValueError):
		load_annotation_volume(resolution_um=7)


def test_load_annotation_volume_returns_array(monkeypatch):
	# Success path: download + NRRD parsing are mocked so no network is hit.
	# The contract is a bare 3D array (the dashboard relies on `.shape`).
	fake_volume = np.zeros((2, 3, 4), dtype=np.uint32)
	monkeypatch.setattr(bg, "download_bytes", lambda url: b"raw-bytes")
	monkeypatch.setattr(bg.nrrd, "read_header", lambda memory_file: {"sizes": [2, 3, 4]})
	monkeypatch.setattr(bg.nrrd, "read_data", lambda header, memory_file: fake_volume)

	out = load_annotation_volume(resolution_um=25)
	assert isinstance(out, np.ndarray)
	assert out.shape == (2, 3, 4)


def test_load_structure_graph_flattens_tree(monkeypatch):
	# A nested ontology (root -> grey -> Isocortex) must flatten to one row per
	# node with parent_structure_id wired from the tree (root's parent is None).
	ontology = {
		"msg": [
			{
				"id": 997,
				"acronym": "root",
				"name": "root",
				"graph_order": 0,
				"structure_id_path": "/997/",
				"color_hex_triplet": "FFFFFF",
				"children": [
					{
						"id": 8,
						"acronym": "grey",
						"name": "Basic cell groups and regions",
						"graph_order": 1,
						"structure_id_path": "/997/8/",
						"color_hex_triplet": "BFDAE3",
						"children": [
							{
								"id": 315,
								"acronym": "Isocortex",
								"name": "Isocortex",
								"graph_order": 2,
								"structure_id_path": "/997/8/315/",
								"color_hex_triplet": "70FF71",
								"children": [],
							}
						],
					}
				],
			}
		]
	}
	monkeypatch.setattr(bg, "download_bytes", lambda url: json.dumps(ontology).encode("utf-8"))

	df = load_structure_graph()

	assert set(df["id"]) == {997, 8, 315}
	for col in (
		"id",
		"acronym",
		"name",
		"parent_structure_id",
		"graph_order",
		"structure_id_path",
		"color_hex_triplet",
	):
		assert col in df.columns

	# Mixing None (root) with ints makes the column float, so root reads as NaN.
	parent = dict(zip(df["id"], df["parent_structure_id"]))
	assert pd.isna(parent[997])
	assert parent[8] == 997
	assert parent[315] == 8


def test_get_slice_view_per_orientation():
	vol = np.arange(2 * 3 * 4).reshape(2, 3, 4)
	assert np.array_equal(get_slice_view(vol, 1, "coronal"), vol[1, :, :])
	assert np.array_equal(get_slice_view(vol, 2, "sagittal"), vol[:, :, 2])
	assert np.array_equal(get_slice_view(vol, 0, "horizontal"), vol[:, 0, :])


def test_get_slice_view_unknown_orientation():
	with pytest.raises(ValueError):
		get_slice_view(np.zeros((2, 2, 2)), 0, "oblique")


def _square(side, offset=0.0):
	o = offset
	return Polygon([(o, o), (o + side, o), (o + side, o + side), (o, o + side)])


def test_clean_polygons_drops_small_components():
	big = _square(10)  # area 100
	tiny = _square(1, offset=20)  # area 1, well separated
	result = clean_polygons_geometry([big, tiny], min_area_px=5.0, simplify_px=0.0)
	assert isinstance(result, MultiPolygon)
	# Only the big square survives the min-area filter.
	assert len(result.geoms) == 1
	assert result.area == pytest.approx(100.0, rel=1e-6)


def test_clean_polygons_repairs_invalid_geometry():
	# Self-intersecting "bow-tie" is invalid; buffer(0) should repair it.
	bowtie = Polygon([(0, 0), (4, 4), (4, 0), (0, 4)])
	assert not bowtie.is_valid
	result = clean_polygons_geometry([bowtie], min_area_px=1.0, simplify_px=0.0)
	assert result is not None
	assert result.is_valid


def test_clean_polygons_empty_input_returns_none():
	assert clean_polygons_geometry([], min_area_px=1.0, simplify_px=0.0) is None


@pytest.mark.parametrize("mode", ["raster", "contour"])
def test_mask_to_polygon_solid_block(mode):
	mask = np.zeros((20, 20), dtype=bool)
	mask[3:17, 3:17] = True
	geom = mask_to_polygon(mask, min_area_px=5.0, simplify_px=0.5, polygon_mode=mode)
	assert isinstance(geom, MultiPolygon)
	assert geom.area > 0


def test_mask_to_polygon_empty_mask_returns_none():
	mask = np.zeros((10, 10), dtype=bool)
	assert mask_to_polygon(mask, min_area_px=5.0, simplify_px=0.5) is None


def test_mask_to_polygon_unknown_mode():
	mask = np.ones((10, 10), dtype=bool)
	with pytest.raises(ValueError):
		mask_to_polygon(mask, min_area_px=5.0, simplify_px=0.5, polygon_mode="bogus")


def test_build_geojson_integration(synthetic_volume, structure_df):
	geojson = build_geojson(
		volume=synthetic_volume,
		structure_df=structure_df,
		orientation="coronal",
		resolution_um=25,
		slice_indices=[1],
		min_area_px=5.0,
		simplify_px=0.5,
		polygon_mode="raster",
		lon_range=(-15.0, 15.0),
		lat_range=(-10.0, 10.0),
	)

	assert geojson["type"] == "FeatureCollection"
	assert len(geojson["features"]) == 1

	feat = geojson["features"][0]
	props = feat["properties"]
	assert props["feature_id"] == "1_315"
	assert props["Region ID"] == 315
	assert props["Region name"] == "Isocortex"
	assert props["slice_index"] == 1
	assert props["orientation"] == "coronal"
	assert feat["geometry"]["type"] == "MultiPolygon"

	# Coordinates were scaled into the requested lon/lat ranges.
	coords = np.array(
		[pt for poly in feat["geometry"]["coordinates"] for ring in poly for pt in ring]
	)
	assert coords[:, 0].min() >= -15.0 - 1e-6
	assert coords[:, 0].max() <= 15.0 + 1e-6
	assert coords[:, 1].min() >= -10.0 - 1e-6
	assert coords[:, 1].max() <= 10.0 + 1e-6


def test_build_geojson_requires_exactly_one_selection(synthetic_volume, structure_df):
	# Both slice_indices and coords_mm -> error.
	with pytest.raises(ValueError):
		build_geojson(
			volume=synthetic_volume,
			structure_df=structure_df,
			orientation="coronal",
			slice_indices=[1],
			coords_mm=[-2.0],
		)
	# No selection method at all -> error.
	with pytest.raises(ValueError):
		build_geojson(
			volume=synthetic_volume,
			structure_df=structure_df,
			orientation="coronal",
		)


def test_save_geojson_preserves_built_coordinates(tmp_path, synthetic_volume, structure_df):
	# build_geojson already scales pixel coords into lon/lat space. Saving must
	# persist that geometry verbatim, not scale (and y-flip) it a second time.
	geojson = build_geojson(
		volume=synthetic_volume,
		structure_df=structure_df,
		orientation="coronal",
		resolution_um=25,
		slice_indices=[1],
		min_area_px=5.0,
		simplify_px=0.5,
		polygon_mode="raster",
		lon_range=(-15.0, 15.0),
		lat_range=(-10.0, 10.0),
	)
	before = copy.deepcopy(geojson["features"][0]["geometry"]["coordinates"])

	out_path = save_geojson(geojson, str(tmp_path / "slice.geojson"))
	with open(out_path, encoding="utf-8") as f:
		reloaded = json.load(f)
	after = reloaded["features"][0]["geometry"]["coordinates"]

	assert _flat_coords(after) == pytest.approx(_flat_coords(before))
