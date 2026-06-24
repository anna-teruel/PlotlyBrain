<div align="center">

<p align="center">
  <img src="docs/logos/GeoBrain_logo1.png" width="75%">
</p>
</div>

GeoBrain is an interactive Python framework for atlas-based visualization of quantitative histological data. It maps region-wise metrics derived from atlas-registered workflows, such as QUINT [1], onto the Allen Mouse Brain Common Coordinate Framework (CCFv3) [2]. Built on Plotly, geobrain provides interactive 2D atlas navigation, group comparisons, customizable color mapping, web-based dashboard for exploratory analysis and publication-quality exports.


## Installation

To install:

```
git clone https://github.com/anna-teruel/geobrain
cd geobrain
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
