# CONTEXT.md

# Build Full GitHub Pages Interface + Repository Structure

This document describes exactly what the final interface, repository, and documentation must contain for the hackathon submission. We must use this information to generate:

1. A GitHub Pages website (`/docs/index.html`, CSS, JS).
2. A full README.md for the GitHub repo.
3. Supporting files like license, requirements, instructions.
4. Narratives, explanations, and all text required by the judges.

Everything must be functional, simple, compliant, and accurate. No fluff. Only the essential content.

## 1. Project Summary

This project builds a machine-learning system that predicts Great Lakes ice concentration for the next 3 days, using:

- NIC ice charts (shapefiles)
- GLSEA surface temperature + ice
- HRRR weather variables
- Bathymetry (GEBCO)
- Shipping route shapefiles
- Land mask from ASC ice files

The trained model is a 3D U-Net that consumes a spatio-temporal stack:

- Input: 3 days of history (T-2, T-1, T0)
- Output: 3 days of forecast (T+1, T+2, T+3)

The GitHub Pages interface displays:

- The initial condition (T0) + 3-day forecast
- Narrative descriptions
- Ice risk overlays: shipping routes, fast-ice vs pack-ice tagging
- Download links for NetCDF + PNGs
- A short explanation of the method

## 2. GitHub Pages Requirements

Must generate:

- File: `/docs/index.html`
- Assets: `/docs/style.css`, `/docs/script.js`

Behavior:

- Show 4 day buttons: T0, T+1, T+2, T+3
- Display forecast images: `forecast_T0.png`, `forecast_T1.png`, `forecast_T2.png`, `forecast_T3.png`
- Show shipping routes overlay (static PNG or vector overlay)
- Perform fast-ice detection using shoreline proximity rule (overlay PNG or computed mask)
- Provide narrative text for each day
- Provide a “How it Works / Methods Summary” section
- Provide a “Download Forecast (NetCDF)” link
- Provide a link back to the GitHub repo
- Provide a small "About the Data" section with references

The interface must be static only (HTML/CSS/JS). No Python runs in the browser.

## 3. Repository Requirements

Must generate a full README.md describing required sections (Overview, Key features, How the model works, Data pipeline summary, Installation instructions, How to reproduce training, How to generate the forecast, How to run the GitHub Page locally, External dataset references).

Required files in repo:

```
README.md
CONTEXT.md
requirements.txt
LICENSE
train.py
run_test_forecast.py
visualize.py
dataset.py
data_loaders.py
data_loaders_test.py
model.py
utils.py or utilities.py
checkpoints/best_model.pth
forecasts/submission_forecast_T0_to_T3.nc
forecasts/forecast_T0.png
forecasts/forecast_T1.png
forecasts/forecast_T2.png
forecasts/forecast_T3.png
docs/index.html
docs/style.css
docs/script.js
```

All filenames and paths must match.

## 4. Narrative Requirements

Must embed narratives into the UI and README. Daily forecast narratives for T0 through T+3. Solution narrative describing model inputs, normalization, reprojection pipeline, cold-start strategy, overlapping patch inference (stride = 128), land-masked loss. Improvements over USNIC and future work list.

## 5. Required Insight Layers

- Shipping route overlay
- Fast Ice Detection (ice_conc > 30% AND distance_to_land < threshold)
- Pack ice flagging optional (text acceptable)

## 6. Description of All External Data Sources

Include references for NIC Ice Charts, GLSEA, HRRR, GEBCO bathymetry, shipping routes shapefile, ASC ice concentration files, plus provided additional NOAA/GLERL resources.

## 7. Evaluation Criteria — Must Satisfy All

Explain potential impact, technological implementation, design, and quality of idea in README and site.

## 8. UI Content Checklist

UI must include: title + subtitle, 4 forecast images, buttons for days, narrative for each day, download NetCDF link, shipping route overlay, fast ice overlay, “How it Works,” “Data Sources,” “Model Architecture,” “Operational Relevance,” link to GitHub repository, license/contact/disclaimers.

## 9. README.md Structure

1. Title + one-sentence summary
2. Demo link (GitHub Pages URL placeholder)
3. Project overview
4. Features
5. Data sources (with citations)
6. Model architecture explanation
7. Installation instructions
8. Training instructions
9. Forecast generation instructions
10. Directory structure
11. Limitations & future work
12. License
13. Contact

## 10. Instructions

Produce fully functional docs HTML/CSS/JS, README, MIT license, and ensure UI is copy-paste ready. No Python runs in browser; do not change Python code beyond file additions where required.

