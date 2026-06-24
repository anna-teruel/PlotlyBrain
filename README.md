<div align="center">

<p align="center">
<<<<<<< HEAD
  <img src="docs/plotly_brain_logo.png" width="45%">
</p>
</div>

PlotlyBrain is an interactive Python framework for atlas-based visualization of quantitative histological data. It maps region-wise metrics derived from atlas-registered workflows (such as QUINT) onto the Allen Mouse Brain Common Coordinate Framework, combining measures of signal abundance and cross-animal consistency. Using a Plotly-based interface, PlotlyBrain enables dynamic exploration of brain-wide patterns across regions, rostro–caudal levels, experimental groups, and markers, while also supporting high-resolution static exports and 3D renderings. The framework is modular, marker-agnostic, and designed to improve the interpretability and accessibility of large-scale neuroanatomical datasets.  ￼
=======
  <img src="docs/logos/GeoBrain_logo1.png" width="75%">
</p>
</div>

GeoBrain is an interactive Python framework for atlas-based visualization of quantitative histological data. It maps region-wise metrics derived from atlas-registered workflows, such as QUINT [1], onto the Allen Mouse Brain Common Coordinate Framework (CCFv3) [2]. Built on Plotly, geobrain provides interactive 2D atlas navigation, group comparisons, customizable color mapping, web-based dashboard for exploratory analysis and publication-quality exports.

>>>>>>> main

## Installation

To install:

```
<<<<<<< HEAD
git clone https://github.com/anna-teruel/PlotlyBrain
cd PlotlyBrain
=======
git clone https://github.com/anna-teruel/geobrain
cd geobrain
>>>>>>> main
uv venv
.venv\Scripts\activate
uv pip install .
```

## Documentation

- [Using the dashboard](docs/dashboard_usage.md) - launching the app and the full workflow: load atlas, build slices, compute scores, view and export.
- [Filtering & coloring rendered slices](docs/filtering_and_coloring.md) - how to color regions by score, filter and select which regions stay highlighted, and apply a flat color.

## License

This project is licensed under the MIT License © 2026
Anna Teruel-Sanchis and Konrad Danielewski.

See the [LICENSE](LICENSE) file for details.
