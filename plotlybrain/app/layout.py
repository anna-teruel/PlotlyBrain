import dash_mantine_components as dmc
import plotly.express as px
from dash import dash_table, dcc, html

RESOLUTIONS = [{"label": f"{r} µm", "value": str(r)} for r in (10, 25, 50, 100)]
ORIENTATIONS = [
	{"label": "Coronal", "value": "coronal"},
	{"label": "Sagittal", "value": "sagittal"},
	{"label": "Horizontal", "value": "horizontal"},
]
POLYGON_MODES = [
	{"label": "Contour (smooth)", "value": "contour"},
	{"label": "Raster (fast)", "value": "raster"},
]
SCORES = [
	{"label": "Rel. abundance", "value": "rel_abundance"},
	{"label": "Frequency", "value": "frequency"},
	{"label": "Density", "value": "density"},
]
COLORSCALES = [
	{"label": name, "value": name}
	for name in dir(px.colors.sequential)
	if not name.startswith("_") and isinstance(getattr(px.colors.sequential, name), list)
]
REL_METHODS = [
	{"label": "Within", "value": "within"},
	{"label": "Reference", "value": "reference"},
]
REF_MODES = [
	{"label": "Pooled", "value": "pooled"},
	{"label": "Group", "value": "group"},
]
EXPORT_FORMATS = [{"label": f, "value": f} for f in ("svg", "png", "pdf", "html")]


def _section_title(step: str, title: str):
	return dmc.Group(
		[
			dmc.Badge(step, variant="filled", radius="sm", size="lg"),
			dmc.Text(title, fw=600, size="md"),
		],
		gap="xs",
	)


def _card(children, p="md", **kwargs):
	return dmc.Card(children, withBorder=True, radius="md", shadow="xs", p=p, **kwargs)


CARD_FILL = {"flex": "1 1 auto", "minHeight": 0, "overflowY": "auto"}

# definite column height = viewport minus app chrome (padding 12+12 + header 38+4)
COL_HEIGHT = "calc(100vh - 66px)"


# left panel : processing pipeline
def _step1_load():
	return _card(
		dmc.Tabs(
			[
				dmc.TabsList(
					[
						dmc.TabsTab("Load atlas", value="atlas"),
						dmc.TabsTab("Load processed files", value="processed"),
					]
				),
				dmc.TabsPanel(
					dmc.Stack(
						[
							dmc.Select(
								id="resolution-select",
								label="Resolution",
								data=RESOLUTIONS,
								value="25",
								allowDeselect=False,
							),
							dmc.Button("Load atlas", id="load-raw-btn", fullWidth=True),
							dmc.Group(
								[
									dmc.Loader(
										id="step1-loader", size="sm", style={"display": "none"}
									),
									dmc.Text(id="step1-status", size="sm", c="dimmed"),
								],
								gap="xs",
							),
						],
						gap="sm",
						pt="sm",
					),
					value="atlas",
				),
				dmc.TabsPanel(
					dmc.Stack(
						[
							dcc.Upload(
								id="upload-geojson",
								accept=".geojson,.json",
								multiple=False,
								children=dmc.Button(
									"Load GeoJSON", variant="light", fullWidth=True
								),
							),
							dcc.Upload(
								id="upload-scores",
								accept=".csv",
								multiple=False,
								children=dmc.Button("Load scores", variant="light", fullWidth=True),
							),
							dmc.Text(id="load-cached-status", size="xs", c="dimmed"),
						],
						gap="sm",
						pt="sm",
					),
					value="processed",
				),
			],
			value="atlas",
		),
		style=CARD_FILL,
	)


def _step2_geojson():
	return _card(
		dmc.Stack(
			[
				_section_title("2", "Build slice GeoJSONs"),
				dmc.Select(
					id="orientation-select",
					label="Orientation",
					data=ORIENTATIONS,
					value="coronal",
					allowDeselect=False,
				),
				dmc.Group(
					[
						dmc.NumberInput(
							id="geo-start", label="Start (mm)", value=-2.5, step=0.1, w="31%"
						),
						dmc.NumberInput(
							id="geo-end", label="End (mm)", value=-1.5, step=0.1, w="31%"
						),
						dmc.NumberInput(
							id="geo-step", label="Step (mm)", value=0.5, min=0.0, step=0.05, w="31%"
						),
					],
					grow=True,
					gap="xs",
				),
				dmc.Accordion(
					value=None,
					children=dmc.AccordionItem(
						[
							dmc.AccordionControl("Advanced build options"),
							dmc.AccordionPanel(
								dmc.SimpleGrid(
									[
										dmc.Select(
											id="polygon-mode-select",
											label="Polygon mode",
											data=POLYGON_MODES,
											value="contour",
											allowDeselect=False,
										),
										dmc.NumberInput(
											id="min-area-px",
											label="Min area (px)",
											value=5.0,
											step=1.0,
										),
										dmc.NumberInput(
											id="simplify-px",
											label="Simplify (px)",
											value=0.8,
											step=0.1,
										),
										dmc.NumberInput(
											id="smooth-sigma",
											label="Smooth sigma",
											value=1.0,
											step=0.1,
										),
									],
									cols=2,
									spacing="xs",
									verticalSpacing="xs",
								)
							),
						],
						value="adv",
					),
				),
				dmc.Group(
					[
						dmc.Button(
							"Build GeoJSONs", id="build-geo-btn", disabled=True, style={"flex": 1}
						),
						dmc.Button("Save", id="save-geo-btn", variant="light", disabled=True),
					],
					gap="xs",
				),
				dmc.Progress(id="geo-progress", value=0, animated=False, striped=True),
				dmc.Text(id="geo-progress-label", size="xs", c="dimmed"),
			],
			gap="sm",
		),
		style=CARD_FILL,
	)


def _step3_scores():
	return _card(
		dmc.Stack(
			[
				_section_title("3", "Compute region scores"),
				dmc.Grid(
					[
						dmc.GridCol(
							dmc.Flex(
								[
									dmc.TextInput(
										id="score-data-dir",
										label="QUINT data folder",
										placeholder="path to *_RefAtlasRegions.csv",
										debounce=500,
										style={"flex": 1},
									),
									dmc.Button(
										"Browse",
										id="browse-data-dir-btn",
										variant="light",
									),
								],
								gap="xs",
								align="flex-end",
							),
							span=9,
						),
						dmc.GridCol(
							dmc.TextInput(
								id="score-sep",
								label="Separator",
								value="auto",
								placeholder="auto",
							),
							span=3,
						),
						dmc.GridCol(
							dmc.Flex(
								[
									dmc.TextInput(
										id="metadata-path",
										label="Metadata CSV",
										placeholder="path to metadata csv",
										debounce=500,
										style={"flex": 1},
									),
									dmc.Button(
										"Browse",
										id="browse-metadata-btn",
										variant="light",
									),
								],
								gap="xs",
								align="flex-end",
							),
							span=9,
						),
						dmc.GridCol(
							dmc.TextInput(
								id="metadata-sep",
								label="Separator",
								value=";",
							),
							span=3,
						),
					],
					gutter="xs",
				),
				dmc.Accordion(
					value=None,
					children=dmc.AccordionItem(
						[
							dmc.AccordionControl("Grouping"),
							dmc.AccordionPanel(
								dmc.Stack(
									[
										dmc.SimpleGrid(
											[
												dmc.TextInput(
													id="animal-col",
													label="Animal column",
													value="animal",
												),
												dmc.TextInput(
													id="group-col",
													label="Group column(s)",
													value="group",
												),
												dmc.Select(
													id="rel-method-select",
													label="Rel. abundance method",
													data=REL_METHODS,
													value="within",
													allowDeselect=False,
												),
												dmc.Select(
													id="ref-mode-select",
													label="Reference mode",
													data=REF_MODES,
													value="pooled",
													allowDeselect=False,
												),
											],
											cols=2,
											spacing="xs",
											verticalSpacing="xs",
										),
										dmc.TextInput(
											id="ref-group",
											label="Reference group (if mode=group)",
										),
									],
									gap="xs",
								)
							),
						],
						value="meta",
					),
				),
				dmc.Group(
					[
						dmc.Button(
							"Compute scores",
							id="process-data-btn",
							disabled=True,
							style={"flex": 1},
						),
						dmc.Button("Save", id="save-scores-btn", variant="light", disabled=True),
					],
					gap="xs",
				),
				dmc.Progress(id="score-progress", value=0, animated=False, striped=True),
				dmc.Text(id="score-progress-label", size="xs", c="dimmed"),
			],
			gap="sm",
		),
		style=CARD_FILL,
	)


def _header():
	return dmc.Group(
		[
			dmc.Badge("dashboard", variant="light", color="grape"),
			dmc.Group(
				[
					dmc.Button(
						"Clear",
						id="clear-cache-btn",
						variant="light",
						color="red",
					),
					dmc.Switch(
						id="color-scheme-toggle",
						size="xl",
						checked=True,
						onLabel="🌙",
						offLabel="☀️",
						styles={
							"track": {"cursor": "pointer"},
							"trackLabel": {"fontSize": "1.2rem"},
						},
					),
				],
				gap="sm",
				align="center",
			),
		],
		justify="space-between",
		align="center",
		h=38,
		style={"marginBottom": "4px"},
	)


def _left_panel():
	return dmc.Stack(
		[
			html.Img(
				src="/assets/plotly_brain_logo.png",
				style={
					"width": "100%",
					"maxWidth": "300px",
					"height": "auto",
					"display": "block",
					"margin": "0 auto",
				},
				alt="PlotlyBrain logo",
			),
			_step1_load(),
			_step2_geojson(),
			_step3_scores(),
		],
		gap="md",
		style={"height": COL_HEIGHT},
	)


# right panel: figure, controls, table
def _brain_graph():
	return _card(
		dcc.Graph(
			id="brain-graph",
			style={"height": "100%", "width": "100%"},
			config={
				"displaylogo": False,
				"scrollZoom": True,
				"modeBarButtonsToRemove": ["select2d", "lasso2d"],
			},
		),
		p="xs",
		style={"height": "50vh"},
	)


def _controls_panel():
	return _card(
		dmc.Stack(
			[
				dmc.Text("Slice", fw=600, size="sm"),
				html.Div(
					dcc.Slider(
						id="slice-slider",
						min=0,
						max=0,
						step=1,
						value=0,
						updatemode="drag",
						included=False,
						marks={},
						allow_direct_input=False,
					),
					id="slice-slider-wrap",
				),
				dmc.Text(id="slice-label", size="xs", c="dimmed"),
				dmc.Divider(),
				dmc.Text("Score", fw=600, size="sm"),
				dmc.SegmentedControl(
					id="score-select", data=SCORES, value="rel_abundance", fullWidth=True
				),
				dmc.Select(
					id="group-select", label="Group", data=[], placeholder="—", clearable=False
				),
				dmc.Group(
					[
						dmc.Select(
							id="colorscale-select",
							label="Colorscale",
							data=COLORSCALES,
							value="Viridis",
							style={"flex": 1},
							allowDeselect=False,
						),
						dmc.NumberInput(id="zmin-input", label="zmin", value=-3.0, step=0.5, w=68),
						dmc.NumberInput(id="zmax-input", label="zmax", value=3.0, step=0.5, w=68),
						dmc.Switch(
							id="static-color-toggle",
							checked=True,
							size="sm",
							style={"marginBottom": 8},
						),
						dmc.ColorInput(
							id="static-color",
							label="Flat",
							value="#fa5252",
							format="hex",
							w=110,
						),
					],
					gap="xs",
					align="flex-end",
				),
				dmc.Divider(label="Export", labelPosition="center"),
				dmc.Group(
					[
						dmc.Flex(
							[
								dmc.TextInput(
									id="export-dir",
									label="Output folder",
									value=".",
									style={"flex": 1},
								),
								dmc.Button("Browse", id="browse-export-dir-btn", variant="light"),
							],
							gap="xs",
							align="flex-end",
							style={"flex": 1},
						),
						dmc.TextInput(
							id="export-name", label="File name", value="brain_slice", w=120
						),
						dmc.Select(
							id="export-format",
							label="Format",
							data=EXPORT_FORMATS,
							value="svg",
							w=92,
							allowDeselect=False,
						),
					],
					gap="xs",
					align="flex-end",
					wrap="nowrap",
				),
				dmc.Group(
					[
						dmc.Button("Export current slice", id="export-btn", variant="light"),
						dmc.Button("Export all slices", id="export-all-btn", variant="light"),
					],
					grow=True,
					gap="xs",
				),
				dmc.Progress(id="export-progress", value=0, animated=False, striped=True),
				dmc.Text(id="export-progress-label", size="xs", c="dimmed"),
				dmc.Text(id="export-status", size="xs", c="dimmed"),
			],
			gap="sm",
		),
		style={"height": "100%"},
	)


def _table_panel():
	return _card(
		dmc.Stack(
			[
				dmc.Group(
					[
						dmc.Text(
							"Region scores (current slice) — select rows to keep colored",
							fw=600,
							size="sm",
						),
						dmc.Button(
							"Deselect all",
							id="deselect-all-btn",
							variant="subtle",
							size="compact-xs",
						),
					],
					justify="space-between",
					align="center",
					wrap="nowrap",
				),
				dash_table.DataTable(
					id="results-table",
					columns=[],
					data=[],
					sort_action="native",
					filter_action="native",
					row_selectable="multi",
					page_size=18,
					style_table={"height": "100%", "overflowY": "auto"},
					style_cell={
						"fontFamily": "Inter, system-ui, sans-serif",
						"fontSize": "12px",
						"padding": "4px 8px",
						"textAlign": "left",
						"maxWidth": 160,
						"overflow": "hidden",
						"textOverflow": "ellipsis",
					},
					style_header={"fontWeight": "600", "backgroundColor": "#f1f3f5"},
				),
			],
			gap="xs",
			style={"height": "100%"},
		),
		style={"height": "100%"},
	)


def _right_panel():
	return dmc.Stack(
		[
			_brain_graph(),
			dmc.Flex(
				[
					html.Div(_controls_panel(), style={"flex": "5 1 0", "minWidth": 0}),
					html.Div(_table_panel(), style={"flex": "7 1 0", "minWidth": 0}),
				],
				gap="md",
				style={"flex": "1 1 auto", "minHeight": 0},
			),
		],
		gap="md",
		style={"height": COL_HEIGHT},
	)


def _stores():
	return [
		dcc.Store(id="session-store", storage_type="memory"),
		dcc.Store(id="geometry-store", storage_type="memory"),
		dcc.Store(id="scores-store", storage_type="memory"),
		dcc.Store(id="slices-store", storage_type="memory"),
		dcc.Store(id="colorscale-stops-store", storage_type="memory"),
		dcc.Download(id="download-geojson"),
		dcc.Download(id="download-scores"),
	]


def build_layout():
	return dmc.MantineProvider(
		id="mantine-provider",
		forceColorScheme="dark",
		theme={
			"primaryColor": "indigo",
			"defaultRadius": "md",
			"fontFamily": "Inter, system-ui, sans-serif",
		},
		children=html.Div(
			[
				*_stores(),
				_header(),
				dmc.Grid(
					[
						dmc.GridCol(_left_panel(), span={"base": 12, "md": 3}),
						dmc.GridCol(_right_panel(), span={"base": 12, "md": 9}),
					],
					gutter="md",
				),
			],
			id="app-root",
			style={
				"padding": "12px",
				"backgroundColor": "var(--mantine-color-body)",
				"minHeight": "100vh",
			},
		),
	)
