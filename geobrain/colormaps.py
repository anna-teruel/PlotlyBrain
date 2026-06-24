AURORA: list[str] = [
	"rgb(22, 191, 168)",   # teal
	"rgb(30, 155, 224)",   # cyan-blue
	"rgb(58, 111, 232)",   # blue
	"rgb(107, 83, 222)",   # indigo
	"rgb(154, 76, 214)",   # violet
	"rgb(197, 63, 192)",   # magenta
	"rgb(232, 67, 147)",   # pink
	"rgb(246, 92, 106)",   # rose
	"rgb(250, 126, 78)",   # orange-red
	"rgb(252, 160, 44)",   # orange
	"rgb(255, 210, 63)",   # gold
]

CUSTOM_COLORSCALES: dict[str, list[str]] = {
	"Aurora": AURORA,
}


def resolve_name(name: str | None) -> str | list[str] | None:
	"""Resolve a colorscale name for ``sample_colorscale``.

	Custom names map to their explicit stop lists; any other value is returned
	unchanged so Plotly resolves it as a built-in name. ``sample_colorscale``
	accepts either form, so callers can treat the result opaquely.
	"""
	if name in CUSTOM_COLORSCALES:
		return CUSTOM_COLORSCALES[name]
	return name
