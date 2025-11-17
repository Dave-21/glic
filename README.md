# Great Lakes IceCast

Deterministic 2D U-Net workflow for Great Lakes ice concentration forecasts. The system blends NOAA GLSEA ice/temperature grids, NOAA National Ice Center (NIC) shapefiles, HRRR weather drivers, GEBCO bathymetry, and buffered shipping-route masks into a single training/inference stack.

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

## Masks
- **Land / fast-ice mask**: extracted from GLSEA concentration rasters and ASC land classifications, used for loss weighting, shoreline-fast tagging, and the overlay toggled in `docs/index.html`.
- **Shipping-route corridors**: buffered shapefiles from the [PHMSA NPMS archive](https://www.npms.phmsa.dot.gov/CNWData.aspx) align with U.S. and Canadian corridors and drive the sampling bias plus the dashboard overlay.

See `utilities.get_land_mask` and `utilities.get_shipping_route_mask` for the reproducible preprocessing steps.

## GitHub Pages deployment
The dashboard in `docs/` is ready to serve directly from GitHub Pages. To publish it:

1. Push the latest changes to the `main` branch so that `docs/` holds the compiled assets (`index.html`, `style.css`, `script.js`, `mask_summary.svg`, etc.).
2. In the repository **Settings → Pages** panel, choose `main` as the source branch and `docs/` as the folder, then save. (You already selected this combination—double-check that the dropdown now reads “Deploy from a branch: main / docs”.)
3. Wait for the green “GitHub Pages build and deployment” workflow to finish under **Actions**. You can click the banner that appears on the Pages settings screen to follow the build log.
4. Visit the published URL shown in the Pages panel (e.g., `https://<org>.github.io/<repo>/`). If you see a 404, force-refresh after a minute; first-time builds can take ~2 minutes to propagate.
5. When you update the dashboard, repeat step 1; GitHub Pages automatically rebuilds from the newest `docs/` contents.

Optional: add a custom domain or enforce HTTPS in the same Pages settings page once DNS is configured.

## Source datasets
- **NOAA NIC ice charts (shapefiles)** for situational labels.
- **NOAA GLERL GLSEA surface temperature & ice** ([link](https://www.glerl.noaa.gov/emf/data/yyyy_glsea_ice/)) for concentration grids, land mask, and fast-ice references.
- **NOAA HRRR weather archive on AWS** ([link](https://registry.opendata.aws/noaa-hrrr-pds/)) supplying air temperature, winds, and pressure.
- **GEBCO 2023 bathymetry grid** for static depth context.
- **Shipping route shapefiles** from the [PHMSA NPMS marine corridor dataset](https://www.npms.phmsa.dot.gov/CNWData.aspx).
- **ASC ice concentration + land mask (NOAA GLERL)** to reinforce shoreline classes.
- Optional GLISA + CoastWatch reference archives for manual validation.

## Future work
- Assimilate Sentinel-1/RCM SAR textures to capture shear and ridging not visible in GLSEA.
- Fuse altimetry-derived ice-thickness priors once reliable coverage is available.
- Add lightweight ensemble perturbations to quantify confidence bands for each corridor.

## License
Released under the [MIT License](LICENSE).
