#
# THIS IS THE FULL, CORRECT dataset.py (v9 - Biased Sampling Fix)
#
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import datetime
from pathlib import Path
from typing import List, Dict, Tuple
import xarray as xr
import random
from rasterio.enums import Resampling
import pandas as pd
import re 
import geopandas as gpd 
from rasterio import features 
from rasterio.transform import from_bounds
from concurrent.futures import ThreadPoolExecutor
import json 
from tqdm import tqdm
import warnings # <-- Added for warnings

import config
from config import HRRR_VARS
# --- [FIX 1/5] Corrected Import Order ---
# utilities must be imported BEFORE data_loaders to fix the circular dependency
import utilities
import data_loaders
from data_loaders import get_glsea_ice_data, get_gebco_data
# ----------------------------------------

# --- Dataset Constants ---\
PATCH_SIZE = 256

# --- Channel Definitions ---\
N_INPUT_STATE_CHANNELS = 2    # (Ice_T0, DeltaIce)
N_WEATHER_CHANNELS = len(HRRR_VARS)     # 4 channels
N_GLSEA_CHANNELS = 1          # (Water Temp ONLY)
N_STATIC_CHANNELS = 2         # 1 channel (shipping routes), 1 channel (GEBCO)
N_INPUT_CHANNELS = N_INPUT_STATE_CHANNELS + N_WEATHER_CHANNELS + N_GLSEA_CHANNELS + N_STATIC_CHANNELS
N_OUTPUT_CHANNELS = 1 # (Ice_T1)

# --- [FIX 2/5] Updated GreatLakesDataset Class ---
class GreatLakesDataset(Dataset):
    """
    PyTorch Dataset for loading Great Lakes ice and weather data.
    
    Loads all data into memory on initialization and serves
    randomly selected 2D patches for training.
    """
    
    def __init__(self, is_train=True, weather_stats=None):
        """
        Initializes the dataset, loading all data into RAM.
        
        Args:
            is_train (bool): Flag for training mode. If True, enables
                             biased patch sampling and data augmentation.
            weather_stats (dict, optional): Precomputed normalization stats
                                            (mean, std) for weather data.
                                            If None, stats are calculated.
        """
        print(f"--- Loading GreatLakesDataset (is_train={is_train}) ---")
        self.is_train = is_train
        self.patch_size = PATCH_SIZE
        
        # --- 1. Load Core Data ---
        print("Loading core datasets into memory...")
        # This loads the full 3D+time data (time, y, x)
        self.glsea_data = data_loaders.get_glsea_ice_data()
        
        # This loads the 2D static data (y, x)
        self.gebco_data = data_loaders.get_gebco_data()
        
        # --- 2. Define Master Grid & Masks ---
        print("Loading master grid and static masks...")
        # Get the 2D (y, x) master grid definition
        self.master_grid_2d = utilities.get_master_grid_definition()
        
        # Load 2D (y, x) masks
        self.land_mask = utilities.get_land_mask(self.master_grid_2d)
        self.shipping_mask = utilities.get_shipping_route_mask(self.master_grid_2d)
        
        # --- 3. Calculate Valid Pixel Indices for Sampling ---
        print("Calculating valid sampling indices...")
        
        # Find all pixels that are water (land_mask == 0)
        self.valid_water_indices = np.argwhere(self.land_mask.values == 0)
        
        # Find all pixels that are ALSO shipping routes (shipping_mask == 1)
        self.shipping_route_indices = np.argwhere(
            (self.land_mask.values == 0) & (self.shipping_mask.values == 1)
        )
        
        print(f"Found {len(self.valid_water_indices)} total valid water pixels.")
        print(f"Found {len(self.shipping_route_indices)} valid shipping route pixels for biased sampling.")
        
        if self.is_train and len(self.shipping_route_indices) == 0:
            print("WARNING: No shipping route pixels found! Biased sampling will be disabled.")

        # --- 4. Prepare Static Data Channels ---
        # Combine static channels into one (N_STATIC_CHANNELS, H, W) array
        self.static_data = xr.concat(
            [self.shipping_mask, self.gebco_data],
            dim=pd.Index(["shipping_routes", "gebco"], name="channel")
        ).values.astype(np.float32) # (2, H, W)
        
        # --- 5. Prepare Time-Varying Data ---
        print("Aligning time-varying data...")
        self.available_dates = self._get_available_dates()
        
        # This will hold the (time, 4, H, W) weather data
        self.hrrr_data = None
        
        # --- 6. Handle Weather Data & Normalization ---
        # We pre-load all weather data for the entire date range
        # and calculate stats if they aren't provided.
        print("Loading all weather data (this may take a moment)...")
        self._load_all_hrrr_data()
        
        if weather_stats:
            print("Using provided weather normalization stats.")
            self.weather_stats = weather_stats
        else:
            print("Calculating weather normalization stats...")
            self.weather_stats = self._calculate_weather_stats()
            
        # Normalize the pre-loaded weather data
        self._normalize_hrrr_data()
        
        print(f"--- Dataset loading complete. {len(self)} samples available. ---")

    def _get_available_dates(self) -> List[datetime.date]:
        """
        Gets the list of dates that are valid for training (T-1, T, T+1).
        We need T-1 (for delta), T (for weather), and T+1 (for truth).
        """
        all_dates = pd.to_datetime(self.glsea_data.time.values).date
        
        # We can't use the first or last date
        # T-1 = all_dates[0]  -> T = all_dates[1] -> T+1 = all_dates[2]
        # We need dates from index 1 up to index (len-2)
        valid_dates_T = all_dates[1:-1]
        
        # Store mapping of date -> index in glsea_data
        self.date_to_idx = {date: i for i, date in enumerate(all_dates)}
        
        return list(valid_dates_T)

    def _load_all_hrrr_data(self):
        """
        Loads all HRRR weather data for all available dates into memory.
        """
        all_hrrr_data = []
        print("Pre-loading HRRR data for all dates...")
        for date_T in tqdm(self.available_dates):
            # Load weather data for day T
            hrrr_day = data_loaders.load_hrrr_data_for_day(date_T)
            all_hrrr_data.append(hrrr_day)
            
        # Stack into a single xarray DataArray (time, channel, y, x)
        self.hrrr_data = xr.concat(all_hrrr_data, dim="time")
        self.hrrr_data['time'] = self.available_dates
        self.hrrr_data = self.hrrr_data.load() # Load into memory
        print("HRRR data loaded.")

    def _calculate_weather_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Calculates mean and std for each weather channel
        ONLY over water pixels.
        """
        stats = {}
        # Create a 4D mask (1, 4, H, W)
        mask_4d = np.broadcast_to(
            self.land_mask.values == 0, 
            self.hrrr_data.shape
        )
        
        # Use the mask to select only water pixel values
        valid_data = self.hrrr_data.values[mask_4d]
        
        # Reshape to (Num_Valid_Pixels, Num_Channels)
        # This is tricky. Let's do it per channel.
        for i, var_name in enumerate(HRRR_VARS):
            channel_data = self.hrrr_data.values[:, i, :, :]
            # (T, H, W)
            water_pixels = channel_data[self.land_mask.values == 0]
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean = np.nanmean(water_pixels).astype(np.float32)
                std = np.nanstd(water_pixels).astype(np.float32)
            
            if std < 1e-6:
                std = 1.0
                print(f"Warning: Std dev for {var_name} is near zero. Setting to 1.")
                
            stats[var_name] = {'mean': mean, 'std': std}
            print(f"Stats for {var_name}: mean={mean:.4f}, std={std:.4f}")
            
        return stats

    def _normalize_hrrr_data(self):
        """
        Applies z-score normalization to the pre-loaded HRRR data in-place.
        """
        for i, var_name in enumerate(HRRR_VARS):
            mean = self.weather_stats[var_name]['mean']
            std = self.weather_stats[var_name]['std']
            self.hrrr_data.values[:, i, :, :] = (
                (self.hrrr_data.values[:, i, :, :] - mean) / std
            )
            
    def __len__(self) -> int:
        return len(self.available_dates)

    # --- [FIX 3/5] New Helper for Biased Patch Sampling ---
    def _get_patch_center(self) -> Tuple[int, int]:
        """
        Selects the center pixel (y, x) for a new patch.
        
        If in training mode, it has a 50% chance to select a pixel
        on a shipping route and a 50% chance to select any water pixel.
        """
        # For validation, always pick from any water pixel
        if not self.is_train:
            rand_idx = random.randint(0, len(self.valid_water_indices) - 1)
            return self.valid_water_indices[rand_idx]

        # For training, 50/50 biased sampling
        # (Only if shipping route pixels exist)
        use_shipping_route = (random.random() < 0.5) and (len(self.shipping_route_indices) > 0)
        
        if use_shipping_route:
            # Pick a shipping route pixel
            rand_idx = random.randint(0, len(self.shipping_route_indices) - 1)
            y, x = self.shipping_route_indices[rand_idx]
        else:
            # Pick any valid water pixel
            rand_idx = random.randint(0, len(self.valid_water_indices) - 1)
            y, x = self.valid_water_indices[rand_idx]
            
        return y, x

    # --- [FIX 4/5] Updated __getitem__ Method ---
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Gets a single training/validation sample.
        
        'idx' here corresponds to the time index (which date to use).
        The spatial patch location is chosen randomly.
        """
        
        # --- 1. Select Date (T) ---
        # idx maps to our list of available dates
        date_T = self.available_dates[idx]
        
        # Get the corresponding indices for T-1, T, and T+1
        idx_T = self.date_to_idx[date_T]
        idx_T_minus_1 = idx_T - 1
        idx_T_plus_1 = idx_T + 1
        
        # --- 2. Select Patch Center (y, x) ---
        # This now uses our biased sampling logic
        patch_center_y, patch_center_x = self._get_patch_center()
        
        # --- 3. Define Patch Boundaries ---
        half_patch = self.patch_size // 2
        
        # Calculate slices, clamping to array boundaries
        y_start = max(0, patch_center_y - half_patch)
        y_end = min(self.land_mask.shape[0], y_start + self.patch_size)
        y_start = max(0, y_end - self.patch_size) # Re-adjust start if end was clamped
        
        x_start = max(0, patch_center_x - half_patch)
        x_end = min(self.land_mask.shape[1], x_start + self.patch_size)
        x_start = max(0, x_end - self.patch_size) # Re-adjust start
        
        patch_slice = (slice(y_start, y_end), slice(x_start, x_end))
        
        # --- 4. Extract Input Channels (X) ---
        
        # a) Input State (Ice T-1, Delta Ice)
        ice_T0 = self.glsea_data.isel(time=idx_T).values[patch_slice]
        ice_T_minus_1 = self.glsea_data.isel(time=idx_T_minus_1).values[patch_slice]
        delta_ice = ice_T0 - ice_T_minus_1
        
        # b) Weather (4 channels)
        # We use the index *within the hrrr_data array*, which matches `idx`
        weather_T = self.hrrr_data.values[idx, :, y_start:y_end, x_start:x_end]
        
        # c) GLSEA Water Temp (1 channel)
        # We re-use ice_T0 as it contains both ice (0-100) and water temp (< 0)
        # Let's extract *just* water temp.
        # Water temp is negative, ice is positive.
        water_temp_T0 = np.clip(ice_T0, -5, 0) # Clip to valid water temp range
        
        # d) Static Channels (2 channels)
        static_patch = self.static_data[:, y_start:y_end, x_start:x_end]

        # --- 5. Combine all input channels ---
        # (N_INPUT_CHANNELS, H, W)
        x_channels = [
            ice_T0[np.newaxis, ...],             # Channel 0
            delta_ice[np.newaxis, ...],          # Channel 1
            weather_T,                           # Channels 2-5
            water_temp_T0[np.newaxis, ...],      # Channel 6
            static_patch                         # Channels 7-8
        ]
        
        # Note: Need to handle `weather_T` which is already (C, H, W)
        x = np.concatenate(x_channels, axis=0).astype(np.float32)

        # --- 6. Extract Target (Y) ---
        # Target is ice concentration at T+1
        y = self.glsea_data.isel(time=idx_T_plus_1).values[patch_slice]
        y = y[np.newaxis, ...].astype(np.float32) # (1, H, W)
        
        # --- 7. Extract Land Mask ---
        mask = self.land_mask.values[patch_slice]
        mask = mask[np.newaxis, ...].astype(np.float32) # (1, H, W)

        # --- 8. Data Augmentation (Training only) ---
        if self.is_train:
            x, y, mask = self._augment(x, y, mask)

        # --- 9. Convert to Tensors ---
        return {
            'date': str(date_T),
            'x': torch.from_numpy(x),
            'y': torch.from_numpy(y),
            'land_mask': torch.from_numpy(mask)
        }
        
    def _augment(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Applies random horizontal/vertical flips."""
        
        # Horizontal flip
        if random.random() < 0.5:
            x = np.ascontiguousarray(x[..., ::-1])
            y = np.ascontiguousarray(y[..., ::-1])
            mask = np.ascontiguousarray(mask[..., ::-1])
            
        # Vertical flip
        if random.random() < 0.5:
            x = np.ascontiguousarray(x[..., ::-1, :])
            y = np.ascontiguousarray(y[..., ::-1, :])
            mask = np.ascontiguousarray(mask[..., ::-1, :])
            
        return x, y, mask

# --- [FIX 5/5] Updated __main__ for easier debugging ---
if __name__ == "__main__":
    print("--- Running dataset.py in debug mode ---")
    
    try:
        train_dataset = GreatLakesDataset(is_train=True)
        weather_stats = train_dataset.weather_stats
        
        # Save stats for use by validation
        stats_path = "weather_stats.json"
        print(f"\n--- Saving stats to {stats_path} ---")
        # Need to convert numpy types to native Python types for JSON
        serializable_stats = {}
        for var, values in weather_stats.items():
            serializable_stats[var] = {
                'mean': float(values['mean']),
                'std': float(values['std'])
            }
        with open(stats_path, 'w') as f:
            json.dump(serializable_stats, f, indent=4)
        
        print("\n--- Loading Validation Data (with stats) ---")
        val_dataset = GreatLakesDataset(is_train=False, weather_stats=weather_stats)
        
        print(f"\nTotal training samples: {len(train_dataset)}")
        print(f"Total validation samples: {len(val_dataset)}")
        
        if len(train_dataset) > 0:
            train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
            print("\n--- Testing Training DataLoader (first 3 batches) ---")
            
            for i, batch in enumerate(train_loader):
                if i >= 3:
                    break
                print(f"\n--- Batch {i+1} ---")
                print(f"  Date: {batch['date'][0]} (and {len(batch['date'])-1} others)")
                print(f"  X shape: {batch['x'].shape}")
                print(f"  Y shape: {batch['y'].shape}")
                print(f"  Mask shape: {batch['land_mask'].shape}")
                
                # --- Quick Stats Check ---
                print(f"  X Min: {batch['x'].min():.2f}, Max: {batch['x'].max():.2f}, Mean: {batch['x'].mean():.2f}")
                print(f"  Y Min: {batch['y'].min():.2f}, Max: {batch['y'].max():.2f}, Mean: {batch['y'].mean():.2f}")
                print(f"  Mask Min: {batch['mask'].min():.2f}, Max: {batch['mask'].max():.2f}, Mean: {batch['mask'].mean():.2f}")

        print("\n--- dataset.py debug run complete. ---")

    except Exception as e:
        print(f"\n--- FATAL ERROR during dataset.py debug run ---")
        import traceback
        traceback.print_exc()