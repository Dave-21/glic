import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import datetime
from pathlib import Path
from typing import List, Dict, Tuple
import xarray as xr
import random
from rasterio.enums import Resampling

import config
import data_loaders
import utilities

# --- Dataset Constants ---
PATCH_SIZE = 256      # We'll randomly crop to this size during training
N_TIMESTEPS = 21      # Total days in training set (2019-01-11 to 2019-01-31)

# --- HRRR Weather Variables to use ---
HRRR_VARS = [
    'air_temp',
    'windu',
    'windv',
    'PRATE_surface',
]

# --- State Variables (what we predict) ---
STATE_VARS = ['ice_conc', 'water_temp']

# --- Channel Definitions (UPDATED) ---
N_INPUT_STATE_CHANNELS = 3                                     # (Ice, Temp, IceClass)
N_WEATHER_CHANNELS = len(HRRR_VARS)                            # 4 channels
N_INPUT_CHANNELS = N_INPUT_STATE_CHANNELS + N_WEATHER_CHANNELS # 7 input channels
N_OUTPUT_CHANNELS = N_INPUT_STATE_CHANNELS                     # 2 output channels

class GreatLakesDataset(Dataset):
    """
    Dataset for auto-regressive, 1-day-ahead forecasting.
    Generates pairs of (State @ T, Weather @ T+1) -> (State @ T+1)
    
    Implements Z-score normalization for weather variables.
    """
    def __init__(self,
                 start_dates: List[datetime.date],
                 is_train: bool,
                 weather_stats: Dict[str, Dict[str, float]] = None):
        """
        Initializes the dataset.
        
        Args:
            start_dates: List of dates to use as the "T" state.
            is_train: If True, enables random patching and calculates weather stats.
            weather_stats: If provided (and is_train=False), uses these stats
                           for normalization to prevent data leakage.
        """
        self.start_dates = start_dates
        self.is_train = is_train
        
        # Load the land mask *as an xarray DataArray* first
        land_mask_da = utilities.get_land_mask()
        if land_mask_da is None:
            raise IOError("Could not load land mask from utilities.")
            
        # Store the land mask as a tensor for the model
        self.land_mask_tensor = torch.from_numpy(land_mask_da.values.astype(np.float32))
        
        # Load all weather data into memory
        self.hrrr_data = self.load_all_hrrr()
        
        # --- NEW: Load all ICECON data into memory ---
        # We pass the land mask as the master_grid definition
        self.icecon_data = data_loaders.load_all_icecon(land_mask_da)
        # --- END NEW ---

        # --- Normalization Stats Handling ---
        if is_train:
            print("Training dataset: Calculating weather normalization stats...")
            self.weather_stats = self.calculate_weather_stats(self.hrrr_data, land_mask_da)
        else:
            if weather_stats is None:
                raise ValueError("Validation dataset must be initialized with 'weather_stats' "
                                 "from the training dataset.")
            print("Validation dataset: Using provided weather stats.")
            self.weather_stats = weather_stats
            
        print(f"Loaded dataset: {len(self.start_dates)} samples, is_train={is_train}")

    def calculate_weather_stats(self,
                                hrrr_data: xr.Dataset,
                                land_mask_da: xr.DataArray) -> Dict[str, Dict[str, float]]:
        """
        Calculates the mean and std for each weather variable,
        considering only water pixels from the training data.
        """
        stats = {}
        # Create a water mask (where land == 0)
        water_mask = (land_mask_da == 0)
        
        # Apply this mask to the entire time-series of weather data
        # This will set all land pixels to NaN
        hrrr_water_only = hrrr_data.where(water_mask)

        for var in HRRR_VARS:
            var_data = hrrr_water_only[var]
            
            # Calculate mean and std, skipping NaNs (land pixels)
            mean = float(var_data.mean(skipna=True))
            std = float(var_data.std(skipna=True))
            
            # --- START FIX (from last step) ---
            # Use 1e-3 to catch the 1e-4 std dev of PRATE
            if std < 1e-3:
                print(f"  > WARNING: {var} std is near-zero ({std:.8f}). Setting to 1.0 to avoid division by zero.")
                std = 1.0
            # --- END FIX ---
                
            stats[var] = {'mean': mean, 'std': std}
            print(f"  > {var}: mean={mean:.4f}, std={std:.4f}")
            
        return stats

    def load_all_hrrr(self) -> xr.Dataset:
        """Loads and re-projects the entire HRRR dataset into memory."""
        print("Loading all HRRR data into memory...")
        with xr.open_dataset(config.TRAIN_HRRR_NC_FILE) as ds:
            rename_dict = {}
            if 'x' not in ds.coords and 'x' not in ds.dims:
                if 'longitude' in ds.coords: rename_dict['longitude'] = 'x'
                elif 'lon' in ds.coords: rename_dict['lon'] = 'x'
            
            if 'y' not in ds.coords and 'y' not in ds.dims:
                if 'latitude' in ds.coords: rename_dict['latitude'] = 'y'
                elif 'lat' in ds.coords: rename_dict['lat'] = 'y'

            if rename_dict:
                ds = ds.rename(rename_dict)
            
            try:
                ds = ds.rio.set_spatial_dims(x_dim='x', y_dim='y')
            except Exception as e:
                print(f"[ERROR] Could not set spatial dims on HRRR file: {e}")
                raise

            if ds.rio.crs is None:
                print("HRRR file has no CRS. Writing default: EPSG:4326")
                ds.rio.write_crs(config.MASTER_GRID_CRS, inplace=True)
            
            ds_vars = ds[HRRR_VARS]
            master_grid = utilities.get_master_grid_definition()
            
            ds_reprojected = ds_vars.rio.reproject_match(
                master_grid, 
                resampling=Resampling.bilinear
            )
            return ds_reprojected.load()

    def __len__(self) -> int:
        return len(self.start_dates)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # --- 1. Get Dates ---
        date_T = self.start_dates[idx]
        date_T_plus_1 = date_T + datetime.timedelta(days=1)
        
        # --- 2. Load State @ T (Input) ---
        ice_T_raw_da = data_loaders.load_ice_asc(date_T)
        temp_T_raw_da = data_loaders.load_glsea_water_temp(date_T)
        
        # --- NEW: Load IceClass @ T (Input) ---
        ice_class_T_raw = self.icecon_data['ice_class'].sel(
            time=date_T.strftime('%Y-%m-%d'),
            method='nearest'
        )
        # --- END NEW ---

        # --- 3. Load Weather @ T+1 (Input) ---
        weather_T_plus_1_raw = self.hrrr_data.sel(
            time=date_T_plus_1.strftime('%Y-%m-%d'),
            method='nearest'
        )
        
        # --- 4. Load State @ T+1 (Target) ---
        ice_T_plus_1_raw_da = data_loaders.load_ice_asc(date_T_plus_1)
        temp_T_plus_1_raw_da = data_loaders.load_glsea_water_temp(date_T_plus_1)

        # --- 5. Preprocess & Normalize (as numpy) ---

        # ICE: clamp land (-1) to 0, then scale to [0,1]
        ice_T_raw = ice_T_raw_da.values
        ice_T_raw = np.where(ice_T_raw < 0, 0.0, ice_T_raw)
        ice_T_norm = ice_T_raw / 100.0

        ice_T_plus_1_raw = ice_T_plus_1_raw_da.values
        ice_T_plus_1_raw = np.where(ice_T_plus_1_raw < 0, 0.0, ice_T_plus_1_raw)
        ice_T_plus_1_norm = ice_T_plus_1_raw / 100.0

        # TEMPERATURE: keep raw °C, just clean NaNs
        temp_T_norm = np.nan_to_num(temp_T_raw_da.values, nan=0.0)
        temp_T_plus_1_norm = np.nan_to_num(temp_T_plus_1_raw_da.values, nan=0.0)
        
        # --- NEW: ICE CLASS: Normalize 0-5 -> 0.0-1.0 ---
        ice_class_norm = ice_class_T_raw.values.astype(np.float32) / 5.0
        # --- END NEW ---

        # WEATHER @ T+1: Normalize using pre-calculated stats
        weather_vars = []
        for var in HRRR_VARS:
            var_data_raw = weather_T_plus_1_raw[var].values
            stats = self.weather_stats[var]
            var_data_norm = (var_data_raw - stats['mean']) / stats['std']
            var_data_clean = np.nan_to_num(var_data_norm, nan=0.0)
            weather_vars.append(var_data_clean)

        # --- 6. Stack into Tensors (UPDATED) ---
        input_state = torch.from_numpy(
            np.stack([ice_T_norm, temp_T_norm, ice_class_norm], axis=0).astype(np.float32)
        )  # [3, H, W]

        input_weather = torch.from_numpy(
            np.stack(weather_vars, axis=0).astype(np.float32)
        )  # [4, H, W]
        
        x = torch.cat([input_state, input_weather], dim=0)  # [7, H, W]

        y = torch.from_numpy(
            np.stack([ice_T_plus_1_norm, temp_T_plus_1_norm], axis=0).astype(np.float32)
        )  # [2, H, W]

        # --- 7. Apply Patching (for training only) ---
        if self.is_train:
            x, y, land_mask = self.apply_random_patch(x, y, self.land_mask_tensor)
        else:
            land_mask = self.land_mask_tensor # Use the full mask

        return {
            "x": x,
            "y": y,
            "land_mask": land_mask
        }

    # --- THIS IS YOUR NEW FUNCTION ---
    def apply_random_patch(self, x, y, land_mask):
        """Crops a random [P, P] patch, biased toward patches that contain ice."""
        h, w = x.shape[1:]
        th = tw = PATCH_SIZE

        # No patching needed
        if w == tw and h == th:
            return x, y, land_mask

        max_tries = 16
        best_patch = None
        best_ice_fraction = -1.0

        for _ in range(max_tries):
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)

            x_patch = x[:, i:i + th, j:j + tw]
            y_patch = y[:, i:i + th, j:j + tw]
            land_patch = land_mask[i:i + th, j:j + tw]

            # Channel 0 is ice concentration (0–1)
            ice_channel = x_patch[0]
            ice_fraction = (ice_channel > 0.05).float().mean().item()

            # Track the best patch seen so far (in case we never hit the threshold)
            if ice_fraction > best_ice_fraction:
                best_ice_fraction = ice_fraction
                best_patch = (x_patch, y_patch, land_patch)

            # If this patch has enough ice, use it immediately
            if ice_fraction >= 0.05:
                return x_patch, y_patch, land_patch

        # Fallback: use the best patch we saw
        return best_patch[0], best_patch[1], best_patch[2]
    # --- END YOUR NEW FUNCTION ---

# --- Test Script ---
if __name__ == "__main__":
    print("Testing GreatLakesDataset (Auto-Regressive with Normalization)...")
    
    all_dates = [
        datetime.date(2019, 1, 11) + datetime.timedelta(days=i)
        for i in range(N_TIMESTEPS)
    ]
    
    valid_start_dates = all_dates[:-1]
    
    if not valid_start_dates:
        raise ValueError("Not enough data to form a single (T, T+1) pair.")

    train_dates = valid_start_dates[:-3]
    val_dates = valid_start_dates[-3:]
    
    print(f"Total valid pairs: {len(valid_start_dates)}. Training on: {len(train_dates)}, Validating on: {len(val_dates)}")
    
    print("\n--- Loading Training Data (with patches) ---")
    train_dataset = GreatLakesDataset(train_dates, is_train=True)
    
    # --- Get stats from training set ---
    weather_stats = train_dataset.weather_stats
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    
    batch = next(iter(train_loader))
    x, y, land_mask = batch['x'], batch['y'], batch['land_mask']
    
    print(f"\nTrain Batch X shape: {x.shape}")
    print(f"Train Batch Y shape: {y.shape}")
    print(f"Land Mask shape: {land_mask.shape}")
    
    print("\n--- Loading Validation Data (full image) ---")
    # --- Pass stats to validation set ---
    val_dataset = GreatLakesDataset(val_dates, is_train=False, weather_stats=weather_stats)
    
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    batch = next(iter(val_loader))
    x, y, land_mask = batch['x'], batch['y'], batch['land_mask']
    
    print(f"\nVal Batch X shape: {x.shape}")
    print(f"Val Batch Y shape: {y.shape}")
    print(f"Land Mask shape: {land_mask.shape}")
    
    print(f"\nNormalized 'air_temp' mean (should be ~0): {x[:, 2, :, :].mean():.4f}")
    print(f"Normalized 'windu' mean (should be ~0): {x[:, 3, :, :].mean():.4f}")
    print("\nTest Complete.")