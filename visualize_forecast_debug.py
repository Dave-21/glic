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
import data_loaders_test as dlt
from model import UNet
from dataset import N_INPUT_CHANNELS, N_OUTPUT_CHANNELS, HRRR_VARS
import datetime
import matplotlib.pyplot as plt
import cmocean

os.environ['KMP_DUPLICATE_LIB_OK']='True'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STATS_FILE = config.PROJECT_ROOT / "weather_stats.json"

def load_weather_stats_from_json() -> dict:
    """Loads the pre-calculated weather stats from a JSON file."""
    print(f"Loading weather stats from {STATS_FILE}...")
    if not STATS_FILE.exists():
        raise FileNotFoundError(f"Missing {STATS_FILE}. Run train.py first.")
    with open(STATS_FILE, 'r') as f:
        return json.load(f)

def normalize_weather_data(weather_da: xr.Dataset, weather_stats: dict) -> torch.Tensor:
    """Normalizes the HRRR forecast using training set stats."""
    normalized_channels = []
    for var in HRRR_VARS:
        stats = weather_stats[var]
        std = stats.get('std', 1.0)
        if std < 1e-6: std = 1e-6
        normalized_data = (weather_da[var].values - stats.get('mean', 0.0)) / std
        normalized_channels.append(normalized_data)
    return torch.from_numpy(np.stack(normalized_channels, axis=1)).float().to(DEVICE)

def generate_original_visualization(final_forecasts, land_mask, output_dir):
    """Generates the 1x4 visualization."""
    print("\n--- Generating Original Visualization ---")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("4-Day Forecast", fontsize=16)
    for i in range(4):
        ax = axes[i]
        forecast_ice = np.where(land_mask.values == 1, np.nan, final_forecasts[i])
        im = ax.imshow(forecast_ice, cmap='cmo.ice', vmin=0, vmax=100, origin='lower')
        ax.set_title(f"T+{i}")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.05, pad=0.05, label="Ice Concentration (%)")
    plt.savefig(output_dir / "original_forecast_visualization.png")

def run_forecast_for_tuning(model_path, output_dir, n_filters):
    """
    Generates a forecast and performance report for a given model.
    Returns a dictionary of metrics.
    """
    print(f"--- Generating Forecast for Model: {model_path} ---")
    os.makedirs(output_dir, exist_ok=True)
    
    weather_stats = load_weather_stats_from_json()
    
    model = UNet(in_channels=N_INPUT_CHANNELS, out_channels=N_OUTPUT_CHANNELS, n_filters=n_filters).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    master_grid = utilities.get_master_grid()
    land_mask = utilities.get_land_mask_from_test_ice()
    
    ice_T0_norm, temp_T0, _ = dlt.load_test_initial_conditions(master_grid)
    ice_T0_tensor = torch.from_numpy(ice_T0_norm).float().to(DEVICE)
    temp_T0_tensor = torch.from_numpy(temp_T0).float().to(DEVICE)

    hrrr_ds_raw = dlt.load_test_hrrr_forecast_raw()
    hrrr_ds_reproj = utilities.reproject_dataset(hrrr_ds_raw, master_grid, resampling_method=Resampling.average)
    rename_dict = {"air_temp": "air_temp_2m", "windu": "u_wind_10m", "windv": "v_wind_10m", "PRATE_surface": "precip_surface"}
    hrrr_ds_vars = hrrr_ds_reproj[list(rename_dict.keys())].rename(rename_dict)
    
    # Convert time in hours to timedelta and resample to daily means
    hrrr_ds_vars['time'] = pd.to_timedelta(hrrr_ds_vars['time'].values, unit='h')
    hrrr_daily_mean = hrrr_ds_vars.resample(time='1D').mean()
    
    # Select T+1, T+2, T+3 (indices 1, 2, 3)
    hrrr_forecast_days = hrrr_daily_mean.isel(time=slice(1, 4))
    weather_forecast_norm = normalize_weather_data(hrrr_forecast_days, weather_stats)
    
    initial_state_tensor = torch.zeros(N_INPUT_CHANNELS, master_grid.shape[0], master_grid.shape[1], device=DEVICE)
    initial_state_tensor[0, :, :] = ice_T0_tensor
    initial_state_tensor[1, :, :] = torch.zeros_like(ice_T0_tensor) # Assuming zero delta for simplicity in test
    initial_state_tensor[2:6, :, :] = weather_forecast_norm[0, :, :,]
    initial_state_tensor[6, :, :] = temp_T0_tensor
    initial_state_tensor[7, :, :] = torch.from_numpy(utilities.get_shipping_route_mask(master_grid).values).float().to(DEVICE)
    initial_state_tensor[8, :, :] = torch.from_numpy(data_loaders.get_gebco_data(master_grid).values[0]).float().to(DEVICE)

    with torch.no_grad():
        # The flip that corrects orientation issues
        flipped_input = torch.flip(initial_state_tensor.unsqueeze(0), dims=[-2])
        forecast_outputs = model(flipped_input).squeeze(0)
    
    ic_conc_denorm = ice_T0_tensor.cpu().numpy() * 100.0
    final_forecasts = [ic_conc_denorm]
    for i in range(forecast_outputs.shape[0]):
        final_forecasts.append(forecast_outputs[i, :, :].cpu().numpy() * 100.0)

    submission_data = np.stack(final_forecasts, axis=0)[:, np.newaxis, :, :]
    submission_data = np.where(land_mask.values[None, None, :, :] == 1, -1, submission_data)
    submission_data = np.clip(submission_data, -1, 100)
    
    forecast_da = xr.DataArray(submission_data, coords={"day": ["T=0 (Initial)", "T+1 Forecast", "T+2 Forecast", "T+3 Forecast"], "channel": ["ice_conc"], "y": master_grid['y'], "x": master_grid['x']}, dims=["day", "channel", "y", "x"])
    forecast_da.rio.write_crs(master_grid.rio.crs, inplace=True)
    forecast_da.to_netcdf(output_dir / "submission_forecast.nc")
    
    # --- Visualization and Metrics ---
    test_start_date = datetime.date(2019, 3, 2)
    forecast_dates = [test_start_date + datetime.timedelta(days=i) for i in range(1, 4)]
    ground_truth_data = dlt.load_test_ground_truth(forecast_dates)
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 20))
    fig.suptitle("3-Day Forecast Performance Report", fontsize=24)
    metrics = {}

    for i in range(3):
        day = i + 1
        date = forecast_dates[i]
        forecast_ice = final_forecasts[day]
        ground_truth_ice_da = ground_truth_data[date]

        if ground_truth_ice_da is None: continue

        ground_truth_ice = ground_truth_ice_da.values * 100.0
        
        water_pixels = (land_mask.values == 0)
        mse = np.mean((forecast_ice[water_pixels] - ground_truth_ice[water_pixels])**2)
        mae = np.mean(np.abs(forecast_ice[water_pixels] - ground_truth_ice[water_pixels]))
        metrics[f'MAE_T{day}'] = mae
        metrics[f'MSE_T{day}'] = mse
        
        forecast_ice_masked = np.where(land_mask.values == 1, np.nan, forecast_ice)
        ground_truth_ice_masked = np.where(land_mask.values == 1, np.nan, ground_truth_ice)
        
        ax = axes[0, i]
        im = ax.imshow(forecast_ice_masked, cmap='cmo.ice', vmin=0, vmax=100, origin='lower')
        ax.set_title(f"Forecast T+{day}\nMAE: {mae:.2f}, MSE: {mse:.2f}")
        
        ax = axes[1, i]
        im_gt = ax.imshow(ground_truth_ice_masked, cmap='cmo.ice', vmin=0, vmax=100, origin='lower')
        ax.set_title(f"Ground Truth T+{day}")
        
        ax = axes[2, i]
        diff = forecast_ice_masked - ground_truth_ice_masked
        im_diff = ax.imshow(diff, cmap='coolwarm', vmin=-50, vmax=50, origin='lower')
        ax.set_title(f"Difference (Forecast - GT) T+{day}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.colorbar(im, ax=axes[0, :], orientation='horizontal', fraction=0.05, pad=0.05)
    fig.colorbar(im_gt, ax=axes[1, :], orientation='horizontal', fraction=0.05, pad=0.05)
    fig.colorbar(im_diff, ax=axes[2, :], orientation='horizontal', fraction=0.05, pad=0.05)
    plt.savefig(output_dir / "performance_report.png")
    
    generate_original_visualization(final_forecasts, land_mask, output_dir)
    plt.close('all') # Close all figures
    
    return metrics

def main():
    """Original main function, now a wrapper."""
    model_path = config.PROJECT_ROOT / "checkpoints" / "best_model.pth"
    #model_path = config.PROJECT_ROOT / "tuning_results" / "adamw30" / "best_model.pth"
    #model_path = config.PROJECT_ROOT / "saved_checkpoints" / "last_update_before_20_epoch.pth"
    output_dir = config.PROJECT_ROOT / "last_forecasts"
    # n_filters is hardcoded in the original model, so we assume it here.
    # This might need to be loaded from a config if it varies.
    n_filters = 64 
    run_forecast_for_tuning(model_path, output_dir, n_filters)

if __name__ == "__main__":
    main()