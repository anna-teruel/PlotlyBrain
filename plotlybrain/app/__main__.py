import argparse
import os
import sys
import webbrowser
from threading import Timer

# macOS: Dash's diskcache background-callback worker is launched with
# multiprocess.Process, which forks. Forking the multithreaded Flask dev server
# trips Apple's Objective-C fork-safety guard ("+[NSNumber initialize] may have
# been in progress ... Crashing instead"), killing the worker before the
# callback runs — so e.g. "Load atlas" silently does nothing. Disabling the
# guard is the documented workaround; it must be set before any fork, and the
# Werkzeug reloader child inherits this env var.
if sys.platform == "darwin":
	os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")


def main() -> None:
	parser = argparse.ArgumentParser(description="Launch the PlotlyBrain dashboard.")
	parser.add_argument("--host", default="127.0.0.1")
	parser.add_argument("--port", type=int, default=8050)
	parser.add_argument("--debug", action="store_true")
	args = parser.parse_args()

	from plotlybrain.app.server import create_app

	app = create_app()

	# Open the browser once, after the server has had a moment to start. The
	# WERKZEUG_RUN_MAIN guard keeps the reloader from opening a new tab on
	# every hot reload.
	if not os.environ.get("WERKZEUG_RUN_MAIN"):
		url = f"http://{args.host}:{args.port}"
		Timer(1, lambda: webbrowser.open(url)).start()

	app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
	main()
