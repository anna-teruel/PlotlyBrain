"""
Dash application factory for the PlotlyBrain dashboard.

Wires Dash Mantine Components, the local ``assets/`` folder, and a
Diskcache-backed background-callback manager (needed so the long-running
processing steps can report progress to ``dmc.Progress`` bars).
"""

from __future__ import annotations

import os

import dash
import diskcache
from dash import Dash, DiskcacheManager

from plotlybrain.app.layout import build_layout

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".plotlybrain_cache")


def create_app() -> Dash:
	"""Create and configure the Dash app (layout + callbacks registered)."""
	cache = diskcache.Cache(_CACHE_DIR)
	background_manager = DiskcacheManager(cache)

	# dash-mantine-components requires React 18 - https://www.dash-mantine-components.com/migration
	dash._dash_renderer._set_react_version("18.2.0")

	app = Dash(
		__name__,
		assets_folder=ASSETS_DIR,
		background_callback_manager=background_manager,
		title="PlotlyBrain",
		suppress_callback_exceptions=True,
		update_title=None,
	)

	app.layout = build_layout()

	# Imported for the side effect of registering callbacks against `app`.
	from plotlybrain.app import callbacks  # noqa: F401

	callbacks.register_callbacks(app)
	return app
