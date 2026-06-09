"""Entry point for the PlotlyBrain dashboard.

Run with either::

    python -m plotlybrain.app
    plotlybrain-app            # console script (after pip install)
"""

from __future__ import annotations

import argparse


def main() -> None:
	parser = argparse.ArgumentParser(description="Launch the PlotlyBrain dashboard.")
	parser.add_argument("--host", default="127.0.0.1")
	parser.add_argument("--port", type=int, default=8050)
	parser.add_argument("--debug", action="store_true")
	args = parser.parse_args()

	from plotlybrain.app.server import create_app

	app = create_app()
	app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
	main()
