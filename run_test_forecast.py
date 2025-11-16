import torch
import numpy as np
import os
from tqdm import tqdm
import xarray as xr
from rasterio.enums import Resampling
import datetime

import config
import utilities
import data_loaders_test as dlt
from model import UNet
from dataset import (
    GreatLakesDataset, HRRR_VARS, N_INPUT_CHANNELS, # N_INPUT_CHANNELS is now 7
    N_OUTPUT_CHANNELS, PATCH_SIZE
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_weather_stats_from_training_set():
    """
    Initializes a training dataset just to extract the weather stats.
    """
    print("Loading training dataset to extract weather stats...")
    dummy_dates = [datetime.date(2019, 1, 11) + datetime.timedelta(days=i) for i in range(N_TIMESTEPS)] # N_TIMESTEPS from dataset
    
    train_dataset = GreatLakesDataset(dummy_dates, is_train=True)
    
    print("Weather stats extracted successfully.")
    return train_dataset.weather_stats

def normalize_weather_day(hrrr_day_data: xr.Dataset, 
                          stats: dict) -> np.ndarray:
    """
    Normalizes one day of *pre-reprojected* weather data using Z-score stats.
    """
    normalized_vars = []
    for var in HRRR_VARS:
        var_data_raw = hrrr_day_data[var].values
        var_stats = stats[var]
        
        var_data_norm = (var_data_raw - var_stats['mean']) / var_stats['std']
        var_data_clean = np.nan_to_num(var_data_norm, nan=0.0)
        normalized_vars.append(var_data_clean)
        
    return np.stack(normalized_vars, axis=0) # (C_weather, H, W)

def run_patch_inference(model, input_tensor):
    """
    Runs inference by tiling, inferring, and blending patches.
    """
    b, c, h, w = input_tensor.shape
    
    output_tensor = np.zeros((b, N_OUTPUT_CHANNELS, h, w), dtype=np.float32)
    counts_tensor = np.zeros((h, w), dtype=np.float32)
    
    window_1d_y = np.hanning(PATCH_SIZE)
    window_1d_x = np.hanning(PATCH_SIZE)
    blend_window_2d = np.outer(window_1d_y, window_1d_x)
    
    blend_window_3d = np.tile(
        blend_window_2d[None, :, :], (N_OUTPUT_CHANNELS, 1, 1)
    )

    patches = []
    for y in range(0, h, PATCH_SIZE // 2):
        for x in range(0, w, PATCH_SIZE // 2):
            if y + PATCH_SIZE > h or x + PATCH_SIZE > w:
                continue
            patches.append((y, x))
            
    for x in range(0, w, PATCH_SIZE // 2):
        if x + PATCH_SIZE > w: continue
        patches.append((h - PATCH_SIZE, x))
    for y in range(0, h, PATCH_SIZE // 2):
        if y + PATCH_SIZE > h: continue
        patches.append((y, w - PATCH_SIZE))
    patches.append((h - PATCH_SIZE, w - PATCH_SIZE))
    
    patches = sorted(list(set(patches)))
    
    print(f"Starting patch inference ({len(patches)} patches)...")
    pbar = tqdm(patches, desc="Inferring patches")

    with torch.no_grad():
        for y, x in pbar:
            input_patch = input_tensor[:, :, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            y_pred_patch = model(input_patch)
            y_pred_patch_numpy = y_pred_patch.cpu().numpy()
            
            output_tensor[0, :, y:y+PATCH_SIZE, x:x+PATCH_SIZE] += (y_pred_patch_numpy[0] * blend_window_3d)
            counts_tensor[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += blend_window_2d
    
    pbar.close()
    counts_tensor[counts_tensor == 0] = 1
    output_tensor[0] /= counts_tensor[None, :, :]
    
    return output_tensor


def run_test_forecast():
    print("--- Generating Auto-Regressive Test Forecast (2D Model) ---")
    
    # 1. Get Weather Normalization Stats
    weather_stats = get_weather_stats_from_training_set()
    
    # 2. Load Model
    print(f"Loading model from {config.MODEL_PATH}")
    # This will now correctly initialize with in_channels=7
    model = UNet(
        in_channels=N_INPUT_CHANNELS,
        out_channels=N_OUTPUT_CHANNELS
    ).to(DEVICE)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.eval()

    # 3. Load Land Mask
    print("Loading land mask...")
    land_mask = utilities.get_land_mask().values
    
    # 4. Load Test Initial Conditions (T=0)
    print("Loading test set initial conditions (T=0)...")
    ice_T0_norm = dlt.load_test_ice_raw().values
    temp_T0_raw = dlt.load_test_water_temp_raw().values
    
    # This is our auto-regressive state (Ice, Temp)
    current_state_np = np.stack([ice_T0_norm, temp_T0_raw], axis=0) # (2, H, W)
    
    # --- NEW: Create Proxy Ice Class for T=0 ---
    print("Creating proxy ice class for T=0...")
    ice_conc_T0 = current_state_np[0] # Get the 0-1 normalized ice conc
    ice_class_proxy = np.zeros_like(ice_conc_T0)

    ice_class_proxy[ice_conc_T0 == 0] = 1.0 # Class 1 (Water)
    ice_class_proxy[ice_conc_T0 > 0.0] = 2.0 # Class 2 (New Ice)
    ice_class_proxy[ice_conc_T0 > 0.3] = 3.0 # Class 3 (Pancake)
    ice_class_proxy[ice_conc_T0 > 0.7] = 4.0 # Class 4 (Consolidated)

    # Normalize 0-5
    current_ice_class_proxy_norm = ice_class_proxy / 5.0
    # --- END NEW ---
    
    # 5. Load and Pre-process ALL Weather Data
    print("Loading all test HRRR data...")
    hrrr_ds_raw = dlt.load_test_hrrr_forecast_raw()
    
    print("Reprojecting full HRRR forecast to master grid...")
    master_grid = utilities.get_master_grid_definition()
    hrrr_ds_reproj = hrrr_ds_raw[HRRR_VARS].rio.reproject_match(
        master_grid,
        resampling=Resampling.bilinear
    ).load()
    del hrrr_ds_raw
    print("HRRR reprojection complete.")
    
    # 6. Run Auto-Regressive Forecast
    n_forecast_days = 3
    forecast_outputs = []
    
    for i in range(n_forecast_days):
        day_index = i + 1
        print(f"\n--- Forecasting Day T+{day_index} ---")
        
        # --- a. Get weather for T+(i+1) ---
        hr_start = day_index * 24
        hr_end = (day_index + 1) * 24
        
        hrrr_day_data = hrrr_ds_reproj.isel(time=slice(hr_start, hr_end)).mean(dim='time')
        weather_stack_norm = normalize_weather_day(hrrr_day_data, weather_stats) # (4, H, W)
        
        # --- b. Combine state and weather (UPDATED) ---
        # Stack (Ice, Temp) + (IceClass) + (Weather)
        input_stack_np = np.concatenate([
            current_state_np,                 # (2, H, W)
            current_ice_class_proxy_norm[None, :, :], # (1, H, W)
            weather_stack_norm                # (4, H, W)
        ], axis=0) # Total 7 channels
        
        # --- c. Run inference ---
        input_tensor = torch.from_numpy(input_stack_np).unsqueeze(0).to(DEVICE, dtype=torch.float32)
        
        # predicted_state shape: (1, 2, H, W) -> (Ice, Temp)
        predicted_state = run_patch_inference(model, input_tensor)
        
        # --- d. Post-process and update state ---
        predicted_state_np = predicted_state[0] # Shape (2, H, W)
        
        # Clamp ice (channel 0) to [0, 1]
        predicted_state_np[0] = np.clip(predicted_state_np[0], 0.0, 1.0)
        
        # Apply land mask (set land pixels to 0)
        predicted_state_np = predicted_state_np * (1.0 - land_mask)
        
        forecast_outputs.append(predicted_state_np)
        
        # --- e. Update state for next loop (UPDATED) ---
        
        # 1. The model output (Ice, Temp) becomes the new base state
        current_state_np = predicted_state_np

        # 2. Re-calculate the proxy ice class from the *new* prediction
        new_ice_conc = current_state_np[0] # Get 0-1 normalized ice
        current_ice_class_proxy_norm = np.zeros_like(new_ice_conc)
        current_ice_class_proxy_norm[new_ice_conc == 0] = 1.0 / 5.0 # Class 1
        current_ice_class_proxy_norm[new_ice_conc > 0.0] = 2.0 / 5.0 # Class 2
        current_ice_class_proxy_norm[new_ice_conc > 0.3] = 3.0 / 5.0 # Class 3
        current_ice_class_proxy_norm[new_ice_conc > 0.7] = 4.0 / 5.0 # Class 4
        # This new proxy will be used in the next loop's concatenate step
        
    print("\nForecast loop complete.")
    
    # 7. Save Submission File
    print("Assembling final 4-day forecast...")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    ic_conc_denorm = (ice_T0_norm * 100.0)
    
    final_forecasts = [ic_conc_denorm]
    
    for i in range(n_forecast_days):
        ice_forecast_denorm = forecast_outputs[i][0] * 100.0
        final_forecasts.append(ice_forecast_denorm)

    submission_data = np.stack(final_forecasts, axis=0)
    submission_data = submission_data[:, np.newaxis, :, :]
    
    submission_data = np.where(land_mask[None, None, :, :] == 1, -1, submission_data)
    submission_data = np.clip(submission_data, -1, 100)

    utilities.print_da_stats(submission_data, "Final Submission Data (0-100, -1=land)")

    forecast_days_str = ["T=0 (Initial)", "T+1 Forecast", "T+2 Forecast", "T+3 Forecast"]
    
    forecast_da = xr.DataArray(
        submission_data,
        coords={
            "day": forecast_days_str,
            "channel": ["ice_conc"],
            "y": master_grid['y'],
            "x": master_grid['x']
        },
        dims=["day", "channel", "y", "x"],
        name="ice_concentration_forecast"
    )
    
    forecast_da.rio.write_crs(config.MASTER_GRID_CRS, inplace=True)
    
    forecast_ds = forecast_da.to_dataset()
    forecast_ds.to_netcdf(config.FORECAST_FILE)
    
    print("--- Official Forecast Submission Generated ---")
    print(f"File saved to: {config.FORECAST_FILE}")

if __name__ == "__main__":
    run_test_forecast()