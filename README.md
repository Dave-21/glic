# Great Lakes IceCast

Deterministic 2D U-Net workflow for Great Lakes ice concentration forecasts. The project aligns NOAA GLSEA ice/temperature grids, HRRR weather drivers, GEBCO bathymetry, a GLSEA-derived land mask, and buffered shipping-route masks into a single training/inference stack.

## Repository layout
- `dataset.py`, `data_loaders.py`, `utilities.py`: assemble tensors, apply land and shipping-route masks, and expose helper functions used by training and inference scripts.
- `model.py`: U-Net backbone used in `train.py` and `run_test_forecast.py`.
- `docs/`: static dashboard (HTML/CSS/JS) plus supporting notes for GitHub Pages.
- `forecasts/`: placeholder NetCDF/PNG outputs expected by the dashboard. Replace with real forecasts before publishing.

## Environment setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training
```bash
python train.py
```
Edit the constants near the top of `train.py` to point at your data directories before launching. The script loads the `GreatLakesDataset` class, builds overlapping 256×256 patches, applies a land mask inside `masked_loss`, and writes checkpoints under `checkpoints/`.

## Forecast generation
```bash
python run_test_forecast.py \
  --config config.py \
  --hrrr-root /data/hrrr \
  --nic-root /data/nic \
  --glsea-root /data/glsea \
  --shipping-shp data/shipping_routes.shp \
  --output-dir forecasts
```
This script rebuilds the inference stack, writes `submission_forecast_T0_to_T3.nc`, and emits daily PNGs consumed by `docs/index.html`. Update `docs/script.js` if you need to surface additional metadata in the dashboard.

## Mask summary
Land and route masks originate from GLSEA ice concentration rasters and buffered NOAA shipping-route shapefiles. Counts (from the latest processed scene) are:

- Land mask: 989,894 land pixels / 143,682 water pixels.
- Route mask: 78,053 pixels flagged within the buffered corridors.

Refer to `utilities.get_land_mask` and `utilities.get_shipping_route_mask` for the reproducible workflow. Replace the placeholder overlays in `docs/fast_ice_mask.png` and `docs/shipping_routes_overlay.png` with real exports prior to release.

## Data sources
- NOAA National Ice Center charts
- NOAA GLERL GLSEA NetCDF archive
- NOAA HRRR (AWS public dataset)
- GEBCO 2023 bathymetry grid
- Shipping route shapefiles supplied with the challenge dataset

## License
Released under the [MIT License](LICENSE).
