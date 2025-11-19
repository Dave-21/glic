import pandas as pd
import config
import torch
import os
import xarray as xr
import json
import numpy as np
from rasterio.enums import Resampling
import torch.nn.functional as F
from tqdm import tqdm
import data_loaders
import utilities
from model import UNet
from dataset import N_INPUT_CHANNELS, N_OUTPUT_CHANNELS, HRRR_VARS
import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# --- Configuration ---
CHECKPOINT_PATH = "checkpoints/best_model.pth"
STATS_FILE = config.PROJECT_ROOT / "weather_stats.json"
OUTPUT_DIR = config.PROJECT_ROOT / "forecasts"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
H, W = 1024, 1024

# --- STRATEGIC REGIONS ---
STRATEGIC_POINTS = {
    "Lake Superior": {
        "Duluth": (46.77, -92.09), "Whitefish_Bay": (46.75, -84.80), "Soo_Locks": (46.52, -84.45)
    },
    "Lake Michigan": { "Green_Bay": (45.20, -87.80), "Mackinac": (45.82, -84.75) },
    "Lake Huron": { "St_Clair_River": (43.05, -82.42), "Georgian_Bay": (45.20, -81.00) },
    "Lake Erie": { "Western_Basin": (41.70, -83.00), "Buffalo": (42.88, -78.95) }
}

def load_weather_stats_from_json() -> dict:
    if not STATS_FILE.exists():
        print("Warning: Stats file not found. Using dummy stats.")
        return {var: {'mean': 0.0, 'std': 1.0} for var in HRRR_VARS}
    with open(STATS_FILE, 'r') as f:
        return json.load(f)

def normalize_weather_data(weather_da: xr.Dataset, weather_stats: dict) -> torch.Tensor:
    normalized_channels = []
    for var in HRRR_VARS:
        stats = weather_stats[var]
        std = stats.get('std', 1.0)
        if std < 1e-6: std = 1e-6
        normalized_data = (weather_da[var].values - stats.get('mean', 0.0)) / std
        normalized_channels.append(normalized_data)
    return torch.from_numpy(np.stack(normalized_channels, axis=1)).float().to(DEVICE)

def get_roi_slice_from_point(lat, lon, radius_px=15):
    y_center = int((lat - 41.0) / (49.5 - 41.0) * H)
    x_center = int((lon - -92.5) / (-75.5 - -92.5) * W)
    return slice(max(0, y_center - radius_px), min(H, y_center + radius_px)), \
           slice(max(0, x_center - radius_px), min(W, x_center + radius_px))

def analyze_ice_state(ice_grid, day_label):
    """Generates text analysis based on the actual grid values."""
    report = { "narrative": "", "shipping": "" }
    alerts, shipping_notes = [], []
    
    # The model output might be flipped relative to our coordinate system logic
    # We assume ice_grid passed here is already oriented correctly (North Up)
    
    for lake, points in STRATEGIC_POINTS.items():
        lake_alerts = []
        for name, (lat, lon) in points.items():
            roi = get_roi_slice_from_point(lat, lon)
            conc = np.mean(ice_grid[roi])
            readable = name.replace("_", " ")
            if conc > 80.0:
                lake_alerts.append(f"{readable} (Heavy)")
                shipping_notes.append(f"BLOCKAGE at {readable}.")
            elif conc > 40.0:
                lake_alerts.append(f"{readable} (Forming)")
                
        if lake_alerts: alerts.append(f"**{lake}:** {', '.join(lake_alerts)}.")

    if not alerts:
        report["narrative"] = "All major waterways open. No significant ice."
        report["shipping"] = "Normal operations."
    else:
        report["narrative"] = f"Critical build-up: {' '.join(alerts)}"
        report["shipping"] = " ".join(shipping_notes) if shipping_notes else "Transit caution advised."

    report["title"] = f"Forecast: {day_label}"
    report["image"] = f"../forecasts/forecast_{day_label}.png"
    return report

def save_transparent_layer(data, filename, color_hex, alpha=0.6):
    plt.figure(figsize=(10, 10), dpi=100)
    c_rgb = mcolors.hex2color(color_hex)
    colors = [(0, 0, 0, 0), (*c_rgb, alpha)] 
    cmap = mcolors.LinearSegmentedColormap.from_list('custom', colors, N=2)
    plt.imshow(data, cmap=cmap, vmin=0, vmax=1, interpolation='nearest', origin='lower')
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    save_path = OUTPUT_DIR / filename
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

def generate_dashboard_assets(final_forecasts, land_mask, shipping_mask):
    """Generates individual PNGs, Overlays, and JS Data."""
    print("\n--- Generating Dashboard Assets ---")
    
    # 1. Save Shipping Overlay
    # Assuming shipping_mask is 1024x1024
    save_transparent_layer(shipping_mask, "shipping_overlay.png", "#f5a623", alpha=0.8)
    
    # 2. Save Forecast Maps (T0-T3)
    # Custom Ice Map: Dark Blue (Water) -> White (Ice)
    colors = [(0.05, 0.1, 0.2), (0.95, 0.95, 1.0)] 
    cmap = mcolors.LinearSegmentedColormap.from_list('IceMap', colors, N=100)
    
    metadata = {}
    
    for i in range(4):
        day_label = f"T{i}"
        data = final_forecasts[i]
        
        # Analyze BEFORE plotting (data is 0-100)
        metadata[day_label] = analyze_ice_state(data, day_label)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        
        # Background (Land)
        ax.imshow(land_mask, cmap='gray', vmin=0, vmax=1, origin='lower')
        
        # Ice (Masked)
        d_masked = np.ma.masked_where(land_mask == 1, data)
        im = ax.imshow(d_masked, cmap=cmap, vmin=0, vmax=100.0, origin='lower')
        
        ax.axis('off')
        
        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label('Ice Concentration (%)', rotation=270, labelpad=15)
        
        plt.savefig(OUTPUT_DIR / f"forecast_{day_label}.png", bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"Saved forecast_{day_label}.png")

    # 3. Save Metadata JS
    js_content = f"window.forecastData = {json.dumps(metadata, indent=4)};"
    with open(OUTPUT_DIR / "forecast_data.js", "w") as f:
        f.write(js_content)
    print("Saved forecast_data.js")

def main():
    print(f"--- Generating FINAL SUBMISSION (Device: {DEVICE}) ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    weather_stats = load_weather_stats_from_json()
    
    # Load Model
    model = UNet(in_channels=N_INPUT_CHANNELS, out_channels=N_OUTPUT_CHANNELS, n_filters=64).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print("Model loaded.")
    else:
        print("Warning: Checkpoint not found.")
    model.eval()

    # --- DATA LOADING (Using your working logic) ---
    print("Loading and processing Test Data...")
    master_grid = utilities.get_master_grid()
    
    land_mask_da = utilities.get_land_mask_from_test_ice()
    land_mask = land_mask_da.values # Numpy
    
    ice_T0_norm, temp_T0, _ = data_loaders.load_test_initial_conditions(master_grid)
    ice_T0_tensor = torch.from_numpy(ice_T0_norm).float().to(DEVICE)
    temp_T0_tensor = torch.from_numpy(temp_T0).float().to(DEVICE)

    hrrr_ds_raw = data_loaders.load_test_hrrr_forecast_raw()
    hrrr_ds_reproj = utilities.reproject_dataset(hrrr_ds_raw, master_grid, resampling_method=Resampling.average)
    rename_dict = {"air_temp": "air_temp_2m", "windu": "u_wind_10m", "windv": "v_wind_10m", "PRATE_surface": "precip_surface"}
    hrrr_ds_vars = hrrr_ds_reproj[list(rename_dict.keys())].rename(rename_dict)
    
    hrrr_ds_vars['time'] = pd.to_timedelta(hrrr_ds_vars['time'].values, unit='h')
    hrrr_daily_mean = hrrr_ds_vars.resample(time='1D').mean()
    hrrr_forecast_days = hrrr_daily_mean.isel(time=slice(1, 4))
    weather_forecast_norm = normalize_weather_data(hrrr_forecast_days, weather_stats)
    
    shipping_mask = utilities.get_shipping_route_mask(master_grid).values
    gebco_data = data_loaders.get_gebco_data(master_grid).values[0]

    # Build Input Tensor
    initial_state_tensor = torch.zeros(N_INPUT_CHANNELS, master_grid.shape[0], master_grid.shape[1], device=DEVICE)
    initial_state_tensor[0, :, :] = ice_T0_tensor
    initial_state_tensor[1, :, :] = torch.zeros_like(ice_T0_tensor) 
    initial_state_tensor[2:6, :, :] = weather_forecast_norm[0, :, :,]
    initial_state_tensor[6, :, :] = temp_T0_tensor
    initial_state_tensor[7, :, :] = torch.from_numpy(shipping_mask).float().to(DEVICE)
    initial_state_tensor[8, :, :] = torch.from_numpy(gebco_data).float().to(DEVICE)

    print("Running inference...")
    with torch.no_grad():
        # Your flip logic
        flipped_input = torch.flip(initial_state_tensor.unsqueeze(0), dims=[-2])
        forecast_outputs = model(flipped_input).squeeze(0)
    
    # Prepare Final Arrays (0-100 scale)
    ic_conc_denorm = ice_T0_tensor.cpu().numpy() * 100.0
    final_forecasts = [ic_conc_denorm]
    for i in range(forecast_outputs.shape[0]):
        final_forecasts.append(forecast_outputs[i, :, :].cpu().numpy() * 100.0)

    # --- 1. Save NetCDF (The Requirement) ---
    print("Saving NetCDF...")
    submission_data = np.stack(final_forecasts, axis=0)[:, np.newaxis, :, :] # [Time, 1, H, W]
    # Apply mask (set land to -1 or NaN as per requirements, usually NaN is safer for viewing)
    submission_data = np.where(land_mask[None, None, :, :] == 1, -1, submission_data)
    submission_data = np.clip(submission_data, -1, 100)
    
    # Load time coordinates from the test weather data file
    ds_weather_test = xr.open_dataset(config.TEST_HRRR_NC, decode_times=False)
    submission_times = ds_weather_test['time'].values
    submission_time_attrs = ds_weather_test['time'].attrs
    n_times = submission_data.shape[0]

    forecast_da = xr.DataArray(
        submission_data.squeeze(1), # [Time, H, W]
        coords={
            "time": submission_times[:n_times],
            "y": master_grid.coords['y'],
            "x": master_grid.coords['x']
        },
        dims=["time", "y", "x"],
        name="ice_concentration",
        attrs={"units": "%"}
    )
    forecast_da.rio.write_crs(config.MASTER_GRID_CRS, inplace=True)
    forecast_da.to_netcdf(OUTPUT_DIR / "submission_forecast.nc")
    print(f"NetCDF saved to {OUTPUT_DIR / 'submission_forecast.nc'}")

    # --- 2. Generate Dashboard Assets ---
    generate_dashboard_assets(final_forecasts, land_mask, shipping_mask)
    print("Done.")

if __name__ == "__main__":
    main()