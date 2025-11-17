# Great Lakes IceCast

A 2D U-Net workflow that fuses NOAA ice, weather, bathymetry, and shipping data to forecast Great Lakes ice concentration for three days ahead.

**Live demo:** https://example.github.io/glic (replace with your GitHub Pages URL once published)

## Overview
Great Lakes IceCast delivers operational ice concentration forecasts for T0 through T+3 to support U.S. Coast Guard and commercial shipping decisions. The system harmonizes NIC ice charts, GLSEA temperature and ice grids, HRRR atmospheric drivers, GEBCO bathymetry, a land mask derived from ASC ice files, and shipping corridors into a single 1 km grid. A channel-stacked 2D U-Net ingests the previous three days of conditions and predicts the next three days, providing both imagery and machine-readable NetCDF output for dispatchers.

## Key Features
- Four-day dashboard with day selectors, narrative summaries, and downloadable NetCDF + PNG assets.
- Shipping route overlay and fast-ice mask derived from shoreline proximity to flag choke points for icebreakers.
- Automated preprocessing pipeline that reprojects, normalizes, and mosaics all sensors onto a unified polar stereographic grid.
- Cold-start friendly inference with overlapping patches (stride 128) to maintain fidelity along coastlines and islands.
- Reproducible training and evaluation scripts tied to checkpoints for rapid experimentation.

## How the Model Works
1. **Data stacking:** NIC shapefile concentrations, GLSEA SST/ice, HRRR weather surfaces, GEBCO bathymetry, shipping routes, and ASC land masks are rasterized onto a common grid and normalized.
2. **Temporal context:** For each target forecast, three historical days (T-2, T-1, T0) are concatenated to create a 3×C×H×W tensor.
3. **2D U-Net:** A depth-wise encoder-decoder with standard 2D convolutions learns spatio-temporal context from the stacked channels with residual skip connections. Loss is masked over land to prevent contamination from missing ocean values.
4. **Inference:** During evaluation, overlapping 256×256 patches are processed with stride 128, predictions are blended, and outputs are clipped to [0, 100] percent ice concentration.
5. **Operational layers:** The predicted ice field is combined with fast-ice tagging (ice > 30% within a shoreline buffer) and optional pack-ice risk flags driven by HRRR winds.

## Data Pipeline Summary
- **Ingest:** Download raw NIC shapefiles, GLSEA NetCDF, HRRR Zarr cubes, GEBCO bathymetry, shipping route shapefiles, and ASC concentration masks.
- **Reprojection:** Use `pyproj`/`rasterio` to reproject every layer into a 1 km Equal-Area grid aligned to Great Lakes tiles.
- **Normalization:** Apply per-variable scaling (z-score for temperature, min-max for concentration) and persist stats in `config.py`.
- **Patch generation:** Assemble sequences of length three, chunk into patches, and serialize training shards.
- **Augmentation:** Random flips, rotations, coarse dropout over open water, and wind-ice consistency checks.
- **Serving:** Convert predictions to NetCDF/PNG, create overlays, and push artifacts to `/forecasts` for GitHub Pages.

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing Training
```bash
python train.py \
  --config config.py \
  --data-root /path/to/training_patches \
  --epochs 150 \
  --batch-size 4 \
  --save-dir checkpoints
```
The script loads the 2D U-Net defined in `model.py`, calls dataset builders from `dataset.py`, and writes checkpoints (including `checkpoints/best_model.pth`).

## Generating a Forecast
```bash
python run_test_forecast.py \
  --config config.py \
  --hrrr-root /data/hrrr \
  --nic-root /data/nic \
  --glsea-root /data/glsea \
  --shipping-shp data/shipping_routes.shp \
  --output-dir forecasts
```
This reproduces the NetCDF (`submission_forecast_T0_to_T3.nc`) and PNGs (`forecast_T0.png` … `forecast_T3.png`) consumed by the website. Use `visualize.py` to regenerate overlays and PNG storyboards.

> **Binary-safe placeholders:** To keep this template PR-friendly, every `.png` inside `docs/` and `forecasts/` is an ASCII placeholder file and `.gitattributes` forces Git to treat PNGs as UTF-8 text. The GitHub Pages UI automatically falls back to inline SVG previews when these placeholder PNGs are detected. Replace the placeholder files with true binary outputs (and update `.gitattributes` if needed) before sharing operational results.

## Running the GitHub Page Locally
```bash
cd docs
python -m http.server 8000
```
Navigate to `http://localhost:8000` to interact with the static dashboard. All assets load from the `docs` folder while NetCDF downloads reference `../forecasts`.

## Data Sources
- **NOAA National Ice Center (NIC) Ice Charts:** https://usicecenter.gov/Products/ArchiveSearchMulti?table=GLGisKmz&linkChange=gre-two
- **NOAA GLERL GLSEA Ice & Surface Temperature:** https://apps.glerl.noaa.gov/thredds/catalog/glsea_ice_nc/catalog.html
- **NOAA HRRR Weather (AWS Public Dataset):** https://registry.opendata.aws/noaa-hrrr-pds/
- **GEBCO 2023 Bathymetry Grid:** https://www.gebco.net/data_and_products/gridded_bathymetry_data/
- **Shipping Route Shapefiles:** Provided within the competition dataset release.
- **ASC Ice Concentration & Land Mask:** https://apps.glerl.noaa.gov/erddap/griddap/GL_Ice_Concentration_GCS.html
- **Supporting references:** GLSEA archive (https://www.glerl.noaa.gov/emf/data/yyyy_glsea/), GLISA data sources, CoastWatch Great Lakes ice classification, and NOAA/GLERL ERDDAP catalogs.

## Directory Structure
```
.
├── CONTEXT.md
├── LICENSE
├── README.md
├── checkpoints/
│   └── best_model.pth
├── config.py
├── data_loaders.py
├── data_loaders_test.py
├── dataset.py
├── docs/
│   ├── index.html
│   ├── script.js
│   ├── style.css
│   ├── fast_ice_mask.png
│   ├── shipping_routes_overlay.png
│   └── forecast_T{0-3}.png
├── forecasts/
│   ├── forecast_T0.png
│   ├── forecast_T1.png
│   ├── forecast_T2.png
│   ├── forecast_T3.png
│   └── submission_forecast_T0_to_T3.nc
├── model.py
├── requirements.txt
├── run_test_forecast.py
├── scripts/
├── train.py
├── utilities.py
├── visualize.py
└── visualize_v2.py
```

## Limitations & Future Work
- Weather-driven pack-ice risk is currently text-only; future iterations will bring HRRR winds client-side.
- Incorporate SAR-derived ice thickness estimates and classify new/melting/thin/thick regimes.
- Extend architecture with ConvLSTMs or transformers to capture longer memory.
- Deploy as a live operational dashboard with rolling ingestion and alerting.
- Assimilate ice motion vectors to track leads and convergence zones in near real time.

## License
Released under the [MIT License](LICENSE).

## Contact
Questions? Open an issue or email the maintainers at `operations@icecast.dev`.
