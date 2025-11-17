#
# THIS IS THE FULL, CORRECT dataset.py (v8 - GLSEA as truth)
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
import json # <-- Added for saving stats
from tqdm import tqdm

import config
from config import HRRR_VARS # <-- Make sure this import is here
import data_loaders
import utilities
from data_loaders import get_glsea_ice_data, get_gebco_data

# --- Dataset Constants ---
PATCH_SIZE = 256

# --- Channel Definitions ---
N_INPUT_STATE_CHANNELS = 2    # (Ice_T0, DeltaIce)
N_WEATHER_CHANNELS = len(HRRR_VARS)     # 4 channels
N_GLSEA_CHANNELS = 1          # (Water Temp ONLY) # --- MODIFIED ---
N_STATIC_CHANNELS = 2         # 1 channel (shipping routes), 1 channel (GEBCO)
N_INPUT_CHANNELS = N_INPUT_STATE_CHANNELS + N_WEATHER_CHANNELS + N_GLSEA_CHANNELS + N_STATIC_CHANNELS # 9 channels # --- MODIFIED ---
N_OUTPUT_CHANNELS = 3

# --- ADD THIS LINE ---
N_TIMESTEPS = 3               # We predict T+1, T+2, T+3

# --- Sampler Bias ---
BIAS_ICE_PIXELS = True
BIAS_SHIPPING_ROUTES = True
ICE_THRESHOLD_FOR_BIAS = 0.01 # (1%)

class GreatLakesDataset(Dataset):
    def __init__(self, is_train=True, weather_stats: Dict = None):
        self.is_train = is_train
        self.patch_size = PATCH_SIZE
        
        self.start_date = config.START_DATE
        self.end_date = config.END_DATE
        self.all_dates_in_range = [self.start_date + datetime.timedelta(days=x) for x in range((self.end_date - self.start_date).days + 1)]
        
        print("Loading master dataset (is_train={})...".format(self.is_train))
        
        # 1. Load Data (order matters)
        
        # --- MODIFIED ---
        # We no longer load nic_ice_data. GLSEA is now our ground truth for ALL ice and water temp.
        # self.nic_ice_data = data_loaders.get_nic_ice_data(self.all_dates_in_range) 
        # self.master_ice_and_temp_data = data_loaders.get_glsea_ice_data()
        # self.gebco_data = data_loaders.get_gebco_data()
        self.master_ice_and_temp_data = data_loaders.get_glsea_ice_data()
        self.gebco_data = data_loaders.get_gebco_data() # Still needed for a static channel
        print(f"gebco_data shape: {self.gebco_data.sizes}")
        print(f"master_ice_and_temp_data shape: {self.master_ice_and_temp_data.sizes}") # --- MODIFIED ---
        # print(f"nic_ice_data shape: {self.nic_ice_data.sizes}") # --- MODIFIED ---
        
        # master_grid = utilities.get_master_grid_definition() # Keep this, it's needed for shipping routes
        # #self.land_mask = utilities.get_land_mask(self.gebco_data) # Pass the GEBCO data
        # self.land_mask = utilities.get_land_mask(self.master_ice_and_temp_data) # Pass the GLSEA data
        # self.shipping_routes_mask = utilities.get_shipping_route_mask(master_grid)
        master_grid_2d = utilities.get_master_grid_definition()
        #self.land_mask = utilities.get_land_mask(master_grid_2d)
        self.land_mask = utilities.get_land_mask(self.master_ice_and_temp_data)
        self.shipping_routes_mask = utilities.get_shipping_route_mask(master_grid_2d)
        
        # 2. Find valid data ranges
        self.valid_start_dates = self.find_valid_start_dates()
        
        if not self.valid_start_dates:
            raise ValueError("No valid start dates found in the specified range.")
        
        # 3. Handle weather stats AND PRE-LOAD HRRR DATA
        self.weather_stats = weather_stats
        self.hrrr_data_cache = {} # This will hold all 91 days of data
        stats_path = config.PROJECT_ROOT / "weather_stats.json"

        if is_train:
            # --- THIS IS THE NEW, SMARTER LOGIC ---
            if stats_path.exists():
                # 1. Stats file exists, just load it
                print(f"Loading weather stats from cache: {stats_path}")
                with open(stats_path, 'r') as f:
                    self.weather_stats = json.load(f)
                
                # 2. Still need to pre-load data, but *without* calculating stats
                print("Pre-loading HRRR data for training...")
                all_weather_data, _ = self.get_weather_stats(calculate_stats=False)
            
            else:
                # 3. Stats file NOT found. Do the full calculation.
                print("Weather stats cache not found.")
                print("Calculating weather stats and pre-loading HRRR data...")
                all_weather_data, stats = self.get_weather_stats(calculate_stats=True)
                self.weather_stats = stats
                
                # 4. Save stats to disk for next time
                try:
                    with open(stats_path, 'w') as f:
                        json.dump(self.weather_stats, f, indent=4)
                    print(f"Saving weather stats to {stats_path}...")
                except Exception as e:
                    print(f"!!! Warning: Could not save weather stats: {e}")
            
            # --- END NEW LOGIC ---

        elif self.weather_stats is None:
             raise ValueError("Validation dataset requires weather_stats to be provided (from training set).")

        else: # This is the validation set
            print("Validation dataset: Using provided weather stats.")
            # We must *still* load the data for the validation set
            print("Pre-loading HRRR data for validation...")
            all_weather_data, _ = self.get_weather_stats(calculate_stats=False)
        
        # --- Common pre-loading logic for all cases ---
        for ds in all_weather_data:
            date_key = str(pd.to_datetime(ds['time'].values).date())
            self.hrrr_data_cache[date_key] = ds
        print(f"HRRR data pre-loaded into memory cache ({len(self.hrrr_data_cache)} days).")

            
        # 4. Find valid patches
        cache_filename = f"valid_patches_{'train' if is_train else 'val'}_{self.start_date.strftime('%Y%m%d')}_{self.end_date.strftime('%Y%m%d')}.json"
        cache_path = config.PROJECT_ROOT / cache_filename
        
        try:
            if cache_path.exists():
                print(f"Loading valid patches from cache: {cache_path}")
                with open(cache_path, 'r') as f:
                    patches_json = json.load(f)
                self.valid_patches = [{'date': datetime.datetime.strptime(p['date'], '%Y-%m-%d').date(), 'h': p['h'], 'w': p['w']} for p in patches_json]
            else:
                print("Cache not found.")
                self.valid_patches = self.find_valid_patches()
                print(f"Saving valid patches to cache: {cache_path}")
                patches_json = [{'date': p['date'].strftime('%Y-%m-%d'), 'h': p['h'], 'w': p['w']} for p in self.valid_patches]
                with open(cache_path, 'w') as f:
                    json.dump(patches_json, f)
        except Exception as e:
            print(f"!!! Warning: Could not use patch cache. Re-generating. Error: {e}")
            self.valid_patches = self.find_valid_patches()
        
        if not self.valid_patches:
            raise ValueError("No valid patches found for training/validation.")
        
        if is_train:
            print("Saving master sample list to 'master_sample_log.csv'...")
            try:
                patches_df = pd.DataFrame(self.valid_patches)
                patches_df['date'] = patches_df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
                log_path = config.PROJECT_ROOT / "master_sample_log.csv"
                patches_df.to_csv(log_path, index=False)
                print(f"Master sample log saved to {log_path}")
            except Exception as e:
                print(f"!!! Warning: Could not save master sample log: {e}")
                
        print(f"Loaded dataset: {len(self.valid_patches)} total samples, is_train={is_train}")

    def __len__(self):
        return len(self.valid_patches)

    def find_valid_start_dates(self) -> List[datetime.date]:
        """
        Finds all dates (T0) in the range that have data for:
        T-1 (Weather), T0 (Ice), and T+3 (Ice).
        """
        valid_dates = []
        print(f"Filtering valid start dates from {self.start_date} to {self.end_date}...")
        
        # --- MODIFIED ---
        # nic_data = data_loaders.get_nic_ice_data(self.all_dates_in_range)
        # print(f"Total available dates in NIC data: {len(nic_data['time'])}")
        print(f"Total available dates in GLSEA (master) data: {len(self.master_ice_and_temp_data['time'])}")
        
        for t0 in self.all_dates_in_range:
            t_minus_1 = t0 - datetime.timedelta(days=1)
            t1 = t0 + datetime.timedelta(days=1)
            t2 = t0 + datetime.timedelta(days=2)
            t3 = t0 + datetime.timedelta(days=3)
            
            # We need ice data for T-1, T0, T+1, T+2, T+3
            # We need weather data for T-1
            
            # Check for ice data (T-1 to T+3)
            # --- MODIFIED ---
            has_t_minus_1 = self.check_ice_data_exists(t_minus_1)
            has_t0 = self.check_ice_data_exists(t0)
            has_t1 = self.check_ice_data_exists(t1)
            has_t2 = self.check_ice_data_exists(t2)
            has_t3 = self.check_ice_data_exists(t3)

            if has_t_minus_1 and has_t0 and has_t1 and has_t2 and has_t3:
                valid_dates.append(t0)
        
        print(f"Total valid START dates (T-1 to T+3): {len(valid_dates)}")
        return valid_dates

    # --- MODIFIED --- (Renamed function)
    def check_ice_data_exists(self, target_date: datetime.date) -> bool:
        """Checks if GLSEA ice data exists within 12 hours of the target date."""
        try:
            date_pd = pd.to_datetime(target_date)
            # --- MODIFIED ---
            da = self.master_ice_and_temp_data.sel(time=date_pd, method="nearest")
            time_diff = abs((da.time.values - date_pd.to_numpy()) / np.timedelta64(1, 'h'))
            return time_diff <= 12
        except:
            return False

    def get_weather_stats(self, calculate_stats=True) -> Tuple[List[xr.Dataset], Dict]:
        """
        Loads all HRRR data and optionally calculates mean/std.
        Returns the loaded data AND the stats.
        """
        stats = {var: {'all_values': []} for var in HRRR_VARS}
        all_data = [] # <-- This will hold the loaded data
        
        # Get T-1 dates
        days_to_load = sorted(list(set(date - datetime.timedelta(days=1) for date in self.valid_start_dates)))
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_date = {executor.submit(data_loaders.load_hrrr_data_for_day, date): date for date in days_to_load}
            
            for future in tqdm(future_to_date, desc="Gathering weather stats"):
                try:
                    weather_data = future.result()
                    if weather_data is not None:
                        all_data.append(weather_data) # <-- Store the data
                        
                        if calculate_stats:
                            for var_name in HRRR_VARS:
                                var_data = weather_data[var_name].values
                                stats[var_name]['all_values'].append(var_data[np.isfinite(var_data)])
                except Exception as e:
                    date_for_future = future_to_date[future]
                    print(f"!! Warning: Thread failed for date {date_for_future}: {e}")

        if not calculate_stats:
            return all_data, None # Just return the loaded data

        # --- Your existing stats calculation logic is perfect ---
        final_stats = {}
        total_failure = False
        print("\n--- Calculating Final Weather Stats ---")
        for var_name, data in stats.items():
            if not data['all_values']:
                print(f"!!! FATAL ERROR: No HRRR data was successfully loaded for '{var_name}'.")
                final_stats[var_name] = {'mean': 0.0, 'std': 1.0}
                total_failure = True
            else:
                all_v = np.concatenate(data['all_values'])
                if all_v.size == 0:
                    print(f"!!! FATAL ERROR: Concatenated array for '{var_name}' is empty (all NaNs?).")
                    final_stats[var_name] = {'mean': 0.0, 'std': 1.0}
                    total_failure = True
                else:
                    final_stats[var_name] = {
                        'mean': float(np.mean(all_v)),
                        'std': float(np.std(all_v))
                    }
                    print(f"  {var_name}: Mean={final_stats[var_name]['mean']:.2f}, Std={final_stats[var_name]['std']:.2f}")

        if total_failure:
            raise ValueError("Failed to calculate all weather stats. Stopping training.")
            
        return all_data, final_stats

    def find_valid_patches(self) -> List[Dict]:
        """
        Finds all valid (h, w) corners for patch sampling.
        A patch is valid if it is not 100% land.
        If biasing, it must also contain ice OR be on a shipping route.
        """
        H, W = self.patch_size, self.patch_size
        grid_h, grid_w = self.land_mask.shape
        
        # 1. Get the masks as numpy arrays
        land_mask_np = self.land_mask.values
        shipping_routes_np = self.shipping_routes_mask.values
        
        valid_patches = []
        
        for date_t0 in tqdm(self.valid_start_dates, desc="Finding valid patches"):
            # Get T0 ice data
            # --- MODIFIED ---
            ice_data_t0 = self.get_ice_conc_for_day(date_t0)
            if ice_data_t0 is None:
                continue

            # Iterate over the grid with patch_size steps
            for h in range(0, grid_h - H + 1, H):
                for w in range(0, grid_w - W + 1, W):
                        
                    # 2. Check for 100% land
                    land_mask_patch = land_mask_np[h:h+H, w:w+W]
                    if np.all(land_mask_patch == 1):
                        continue # Skip 100% land patches

                    # --- THIS IS THE NEW BIAS LOGIC (OR instead of AND) ---
                    
                    # Check for ice
                    has_ice = False
                    if BIAS_ICE_PIXELS:
                        ice_patch_t0_nan = ice_data_t0.values[h:h+H, w:w+W]
                        ice_patch_t0 = np.nan_to_num(ice_patch_t0_nan, nan=0.0)
                        if np.any(ice_patch_t0 > ICE_THRESHOLD_FOR_BIAS):
                            has_ice = True
                    
                    # Check for shipping route
                    on_shipping_route = False
                    if BIAS_SHIPPING_ROUTES:
                        shipping_patch = shipping_routes_np[h:h+H, w:w+W]
                        if np.any(shipping_patch == 1):
                            on_shipping_route = True

                    # --- End new logic ---

                    # If we are not biasing at all, add any non-land patch
                    if not BIAS_ICE_PIXELS and not BIAS_SHIPPING_ROUTES:
                        valid_patches.append({'date': date_t0, 'h': h, 'w': w})
                    
                    # If we are biasing, at least one of our biases must be met
                    elif (BIAS_ICE_PIXELS and has_ice) or (BIAS_SHIPPING_ROUTES and on_shipping_route):
                        valid_patches.append({'date': date_t0, 'h': h, 'w': w})

        # After the loop, check if we found anything
        if not valid_patches:
            # This is the new error you are seeing
            raise ValueError("No valid patches found for training/validation.")
            
        return valid_patches

    # --- MODIFIED --- (Renamed and logic updated)
    def get_ice_conc_for_day(self, target_date: datetime.date) -> xr.DataArray | None:
        """
        Safely gets the nearest GLSEA data for a target date and extracts
        ONLY the ice concentration.
        """
        try:
            date_pd = pd.to_datetime(target_date)
            # --- MODIFIED ---
            da = self.master_ice_and_temp_data.sel(time=date_pd, method="nearest")
            time_diff = abs((da.time.values - date_pd.to_numpy()) / np.timedelta64(1, 'h'))
            
            if time_diff <= 12:
                da_squeezed = da.squeeze()
                # --- MODIFIED ---
                # ISOLATE ICE: In GLSEA, ice is < 0. Convert to 0-1 fraction.
                ice_conc = xr.where(da_squeezed < 0, -da_squeezed, 0.0)
                return ice_conc
            else:
                return None
        except Exception as e:
            #print(f"Error getting ice data for {target_date}: {e}")
            return None

    # --- MODIFIED --- (Renamed and logic updated)
    def get_water_temp_for_day(self, target_date: datetime.date, h: int, w: int) -> torch.Tensor | None:
        """
        Safely gets the nearest GLSEA data for a target date and extracts
        ONLY the water temperature.
        Returns a (water_temp_tensor) or (None) if data is not found.
        """
        H, W = self.patch_size, self.patch_size
        try:
            date_pd = pd.to_datetime(target_date)
            # --- MODIFIED ---
            da = self.master_ice_and_temp_data.sel(time=date_pd, method="nearest")
            time_diff = abs((da.time.values - date_pd.to_numpy()) / np.timedelta64(1, 'h'))
            
            if time_diff <= 12:
                patch_data = da.squeeze().values[h:h+H, w:w+W]
                patch_data = np.nan_to_num(patch_data, nan=0.0) # Fill NaNs with 0
                
                # --- MODIFIED ---
                # Split into water temperature and ice concentration
                water_temp = np.where(patch_data > 0, patch_data, 0.0)
                
                return torch.from_numpy(water_temp).float().unsqueeze(0)
            else:
                return None
        except Exception as e:
            #print(f"Error getting GLSEA data for {target_date}: {e}")
            return None

    def build_input_tensor(self, target_date, h, w, fill_val=0.0) -> torch.Tensor:
        """
        Helper to build a single [1, H, W] ice tensor for a given date and patch.
        """
        H, W = self.patch_size, self.patch_size
        # --- MODIFIED ---
        da = self.get_ice_conc_for_day(target_date)
        
        if da is None:
            # This can happen for T-1, fill with zeros
            patch = np.full((H, W), fill_val)
        else:
            patch = da.values[h:h+H, w:w+W]
            # Set land (NaN) to the fill_val (0.0)
            patch = np.nan_to_num(patch, nan=fill_val)
            
        return torch.from_numpy(patch).float().unsqueeze(0) # [1, H, W]

    def build_output_tensor(self, target_date, h, w) -> torch.Tensor:
        """
        Helper to build a single [H, W] target tensor.
        """
        H, W = self.patch_size, self.patch_size
        # --- MODIFIED ---
        da = self.get_ice_conc_for_day(target_date)
        
        if da is None:
            print(f"!!! WARNING: No T+1/T+2/T+3 data found for {target_date}, returning zeros.")
            return torch.zeros((H, W))
            
        patch = da.values[h:h+H, w:w+W]
        # Set land (NaN) to 0.0. The loss mask will handle this.
        patch = np.nan_to_num(patch, nan=0.0)
        
        return torch.from_numpy(patch).float() # [H, W]

    def __getitem__(self, idx):
        H, W = self.patch_size, self.patch_size
        sample_info = self.valid_patches[idx]
        day_t0 = sample_info['date']
        h, w = sample_info['h'], sample_info['w']
        
        # --- 1. Build Input Tensor (x) ---
        day_t_minus_1 = day_t0 - datetime.timedelta(days=1)
        
        # --- MODIFIED --- (These now pull from GLSEA ice via helper functions)
        x_ice_t0 = self.build_input_tensor(day_t0, h, w, fill_val=0.0)
        x_ice_t_minus_1 = self.build_input_tensor(day_t_minus_1, h, w, fill_val=0.0)
        x_ice_delta = x_ice_t0 - x_ice_t_minus_1 # [1, H, W]
        
        static_patch = self.shipping_routes_mask.values[h:h+H, w:w+W]
        x_static_tensor = torch.from_numpy(static_patch).float().unsqueeze(0) # [1, H, W]

        # --- MODIFIED ---
        x_water_temp = self.get_water_temp_for_day(day_t0, h, w)
        if x_water_temp is None:
            # You can comment this out to reduce log spam
            # print(f"!!! WARNING: Missing GLSEA data for {day_t0}, using zeros.")
            x_water_temp = torch.zeros((1, H, W), dtype=torch.float32)
        # --- END MODIFIED ---

        #gebco_patch = self.gebco_data.values[h:h+H, w:w+W]
        gebco_patch = self.gebco_data.values[0, h:h+H, w:w+W]
        x_gebco_tensor = torch.from_numpy(gebco_patch).float().unsqueeze(0) # [1, H, W]

        # --- 1e. Build Weather (T-1) ---
        
        # --- THIS IS THE SPEEDUP FIX ---
        # Get data from the pre-loaded cache instead of S3
        try:
            weather_da = self.hrrr_data_cache[str(day_t_minus_1)]
        except KeyError:
            # Fallback for Windows + num_workers > 0, where memory isn't shared
            # This is slower but will still work.
            # print(f"!!! WARNING: Worker cache empty. Loading HRRR for {day_t_minus_1} dynamically.")
            weather_da = data_loaders.load_hrrr_data_for_day(day_t_minus_1)
        # --- END SPEEDUP FIX ---

        if weather_da is None:
            print(f"!!! WARNING: Missing weather data for {day_t_minus_1}, using zeros.")
            weather_patch_normalized = np.zeros((N_WEATHER_CHANNELS, H, W), dtype=np.float32)
        else:
            #print(f"weather_da shape: {weather_da.sizes}") # DEBUG
            weather_patch_xr = weather_da.isel(y=slice(h, h+H), x=slice(w, w+W))
            weather_patch = weather_patch_xr[HRRR_VARS].to_array().values # Shape [4, H, W]
            
            weather_patch_normalized = np.empty_like(weather_patch, dtype=np.float32)
            for i, var_name in enumerate(HRRR_VARS):
                stats = self.weather_stats[var_name]
                mean = stats['mean']
                std = stats['std']
                var_patch_data = weather_patch[i, :, :].astype(np.float32)
                if std > 1e-6:
                    weather_patch_normalized[i, :, :] = (var_patch_data - mean) / std
                else:
                    weather_patch_normalized[i, :, :] = var_patch_data - mean
            weather_patch_normalized = np.nan_to_num(
                weather_patch_normalized, nan=0.0, posinf=0.0, neginf=0.0
            )

        x_weather_tensor = torch.from_numpy(weather_patch_normalized).float() # [4, H, W]

        # --- 1f. Concatenate all inputs ---
        
        # --- FINAL FIX for NaN Loss ---
        # Convert any NaNs from GLSEA, GEBCO, or NIC data to 0.0
        # before concatenating. This is critical for model stability.
        x_ice_t0 = torch.nan_to_num(x_ice_t0, nan=0.0)
        x_ice_delta = torch.nan_to_num(x_ice_delta, nan=0.0)
        # x_weather_tensor is already clean (we used np.nan_to_num)
        x_water_temp = torch.nan_to_num(x_water_temp, nan=0.0) # --- MODIFIED ---
        # x_glsea_ice_conc = torch.nan_to_num(x_glsea_ice_conc, nan=0.0) # --- MODIFIED ---
        # x_static_tensor is clean (uint8)
        x_gebco_tensor = torch.nan_to_num(x_gebco_tensor, nan=0.0)
        # --- END NaN FIX ---

        # --- MODIFIED ---
        x = torch.cat([
            x_ice_t0,          # [1, H, W]
            x_ice_delta,       # [1, H, W]
            x_weather_tensor,  # [4, H, W]
            x_water_temp,      # [1, H, W]
            # x_glsea_ice_conc,  # [1, H, W] (REMOVED)
            x_static_tensor,   # [1, H, W] (shipping routes)
            x_gebco_tensor     # [1, H, W] (GEBCO bathymetry)
        ], dim=0)               # Total Shape: [9, H, W]
        
        # --- 2. Build Target Tensor (y) ---
        day_t1 = day_t0 + datetime.timedelta(days=1)
        day_t2 = day_t0 + datetime.timedelta(days=2)
        day_t3 = day_t0 + datetime.timedelta(days=3)
        
        # --- MODIFIED --- (These now pull from GLSEA ice via helper functions)
        y1 = self.build_output_tensor(day_t1, h, w) # [H, W]
        y2 = self.build_output_tensor(day_t2, h, w) # [H, W]
        y3 = self.build_output_tensor(day_t3, h, w) # [H, W]
        
        y = torch.stack([y1, y2, y3], dim=0) # Shape [3, H, W]
        
        # --- 3. Build Land Mask ---
        land_mask_patch = self.land_mask.values[h:h+H, w:w+W]
        land_mask_tensor = torch.from_numpy(land_mask_patch).float() # [H, W]

        return {
            'x': x,
            'y': y,
            'land_mask': land_mask_tensor,
            'date': str(day_t0)
        }

# --- Main test block (for debugging) ---
if __name__ == "__main__":
    print("--- Running dataset.py in debug mode ---")
    train_dataset = GreatLakesDataset(is_train=True)
    weather_stats = train_dataset.weather_stats
    print("\n--- Loading Validation Data (with stats) ---")
    val_dataset = GreatLakesDataset(is_train=False, weather_stats=weather_stats)
    
    print(f"\nTotal training samples: {len(train_dataset)}")
    
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
            print(f"  Y NaNs: {torch.isnan(batch['y']).sum()}")
            print(f"  X NaNs: {torch.isnan(batch['x']).sum()}") # Fixed typo here

    print("\n--- dataset.py debug run complete ---")