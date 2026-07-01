"""Tests for I/O helpers: GeoJSON / score loading and figure export."""

import json
import os

import pandas as pd
import plotly.graph_objects as go
import pytest

from geobrain.io import load_geojson, load_score, save_figure


def test_load_score_roundtrip(tmp_path):
	src = pd.DataFrame({"Region ID": [315, 672], "density": [0.3, 0.2]})
	path = tmp_path / "scores.csv"
	src.to_csv(path, index=False)
	out = load_score(str(path))
	pd.testing.assert_frame_equal(out, src)


def test_load_geojson_roundtrip(tmp_path, sample_geojson):
	path = tmp_path / "slice.geojson"
	path.write_text(json.dumps(sample_geojson), encoding="utf-8")
	out = load_geojson(str(path))
	assert out["type"] == "FeatureCollection"
	assert len(out["features"]) == len(sample_geojson["features"])


def test_save_figure_html(tmp_path):
	fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1]))
	out_path = save_figure(fig, str(tmp_path), filename="fig", extension="html")
	assert os.path.isabs(out_path)
	assert out_path.endswith("fig.html")
	assert os.path.exists(out_path)


def test_save_figure_creates_missing_dir(tmp_path):
	fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1]))
	nested = tmp_path / "a" / "b"
	out_path = save_figure(fig, str(nested), filename="fig", extension="html")
	assert os.path.exists(out_path)


def test_save_figure_image(tmp_path):
	# Static image export needs the kaleido backend (and a browser); skip
	# gracefully if the environment can't render.
	pytest.importorskip("kaleido")
	fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1]))
	try:
		out_path = save_figure(fig, str(tmp_path), filename="fig", extension="svg")
	except Exception as exc:  # pragma: no cover - environment dependent
		pytest.skip(f"kaleido image export unavailable: {exc}")
	assert out_path.endswith("fig.svg")
	assert os.path.exists(out_path)
