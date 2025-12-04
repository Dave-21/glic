import datetime
import json
from typing import Dict, List
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
from functools import lru_cache
import logging
import matplotlib.pyplot as plt

import config
import data_loaders
import utilities
from config import HRRR_VARS

# --- Dataset Constants ---
PATCH_SIZE = 256  # The data is pre-processed into grids, but we still sample patches
N_INPUT_STATE_CHANNELS = 2
N_WEATHER_CHANNELS = len(HRRR_VARS)
N_GLSEA_CHANNELS = 1
N_STATIC_CHANNELS = 2
N_CFDD_CHANNELS = 1
N_FLOE_CHANNELS = 1
N_EDGE_CHANNELS = 1
N_INPUT_CHANNELS = (
    N_INPUT_STATE_CHANNELS + N_WEATHER_CHANNELS + N_GLSEA_CHANNELS + N_STATIC_CHANNELS + N_CFDD_CHANNELS + N_FLOE_CHANNELS + N_EDGE_CHANNELS
)
N_OUTPUT_CHANNELS = 3  # Predict all 3 days for both conc & thick
N_TIMESTEPS = 3


class GreatLakesDataset(Dataset):
    def __init__(self, is_train=True, weather_stats: Dict = None):
        if config.DEBUG_MODE:
            print(f"DEBUG: Initializing GreatLakesDataset with is_train={is_train}, weather_stats provided: {weather_stats is not None}")
        self.is_train = is_train
        self.patch_size = PATCH_SIZE
        self.master_grid = utilities.get_master_grid()

        self.start_date = config.START_DATE
        self.end_date = config.END_DATE

        print("Loading master dataset (is_train={})...".format(self.is_train))

        self.land_mask = utilities.get_land_mask(self.master_grid)
        self.shipping_routes_mask = utilities.get_shipping_route_mask(self.master_grid)
        self.gebco_data = data_loaders.get_gebco_data(self.master_grid)

        print("Opening GLSEA and HRRR data for lazy loading...")
        self.glsea_data = data_loaders.load_glsea_ice_data()
        self.hrrr_data = data_loaders.get_consolidated_hrrr_dataset()
        
        print("Loading CFDD data...")
        if config.TRAIN_CFDD_NC_FILE.exists():
            self.cfdd_data = xr.open_dataset(config.TRAIN_CFDD_NC_FILE)["cfdd"]
            # self.cfdd_data.load() # Keep lazy if large, but it's 1024x1024x150 floats ~ 600MB. 
            # Loading into RAM is faster for training.
            print("Loading CFDD into memory for speed...")
            self.cfdd_data.load()
        else:
            print("!!! WARNING: CFDD file not found. Run precompute_cfdd.py.")
            self.cfdd_data = None
            
        print("GLSEA, HRRR, and CFDD data ready.")

        self.valid_start_dates = self.find_valid_start_dates()
        if not self.valid_start_dates:
            raise ValueError("No valid start dates found. Run preprocess_nic_data.py.")

        self.weather_stats = self._get_or_calculate_weather_stats(weather_stats)

        # The "patches" are now just sampling coordinates on the full grid
        self.patches = self.generate_sampling_patches()
        if not self.patches:
            raise ValueError("No valid patches with ice or shipping routes found.")

        print(f"Loaded dataset: {len(self.patches)} total samples, is_train={is_train}")

    def __len__(self):
        return len(self.patches)

    def find_valid_start_dates(self) -> List[datetime.date]:
        """Finds all dates (T0) that have all required pre-processed data."""
        valid_dates = []
        all_possible_dates = [
            self.start_date + datetime.timedelta(days=x)
            for x in range((self.end_date - self.start_date).days + 1)
        ]

        print("Filtering valid start dates based on pre-processed NIC data...")
        for t0 in all_possible_dates:
            required_dates = [
                t0 + datetime.timedelta(days=i) for i in range(-1, N_TIMESTEPS + 1)
            ]
            if all(self.check_nic_data_exists(d) for d in required_dates):
                valid_dates.append(t0)

        print(f"Found {len(valid_dates)} valid start dates.")
        return valid_dates

    def check_nic_data_exists(self, target_date: datetime.date) -> bool:
        """Checks if a pre-processed NIC NetCDF file exists for the given date."""
        return (
            config.NIC_PROCESSED_DIR / f"NIC_{target_date.strftime('%Y-%m-%d')}.nc"
        ).exists()

    def _get_or_calculate_weather_stats(self, provided_stats):
        stats_path = config.PROJECT_ROOT / "weather_stats.json"
        if provided_stats:
            print("Using provided weather stats.")
            return provided_stats
        if stats_path.exists():
            print(f"Loading weather stats from cache: {stats_path}")
            with open(stats_path, "r") as f:
                return json.load(f)

        print("Calculating weather stats from pre-loaded HRRR data...")
        final_stats = {}
        for var_name in HRRR_VARS:
            all_values = self.hrrr_data[var_name].values
            final_stats[var_name] = {
                "mean": float(np.nanmean(all_values)),
                "std": float(np.nanstd(all_values)),
            }
            if (
                np.isnan(final_stats[var_name]["std"])
                or final_stats[var_name]["std"] == 0
            ):
                raise ValueError(
                    f"Stat calculation for '{var_name}' resulted in std=0 or NaN."
                )

        if self.is_train:
            with open(stats_path, "w") as f:
                json.dump(final_stats, f, indent=4)
            print(f"Saved weather stats to {stats_path}")
        return final_stats

    def generate_sampling_patches(self) -> List[Dict]:
        """Generates a list of valid (date, h, w) coords for patch sampling."""
        H, W = self.patch_size, self.patch_size
        grid_h, grid_w = self.land_mask.shape
        patches = []
        for date_t0 in tqdm(
            self.valid_start_dates, desc="Finding valid sampling patches"
        ):
            ds = self.get_ice_data_for_day(date_t0)
            if ds is None:
                continue

            ice_conc_t0 = ds["ice_concentration"].values
            for h in range(0, grid_h - H, 96):  # Stride for efficiency
                for w in range(0, grid_w - W, 96):
                    if np.all(self.land_mask.values[h: h + H, w: w + W]):
                        continue

                    has_ice = np.any(ice_conc_t0[h: h + H, w: w + W] > 0.01)
                    on_shipping_route = np.any(
                        self.shipping_routes_mask.values[h: h + H, w: w + W]
                    )
                    if has_ice or on_shipping_route:
                        patches.append({"date": date_t0, "h": h, "w": w})
        if config.DEBUG_MODE:
            print(f"DEBUG: Generated {len(patches)} sampling patches.")
        return patches

    @lru_cache(maxsize=32)  # Cache up to 32 NIC files per process (Optimized for 16GB RAM)
    def _cached_get_ice_data_for_day(self, target_date_str: str) -> xr.Dataset | None:
        """Loads a pre-processed NIC NetCDF file for a given date from disk."""
        # target_date = datetime.datetime.strptime(target_date_str, "%Y-%m-%d").date() # Not used
        fpath = config.NIC_PROCESSED_DIR / f"NIC_{target_date_str}.nc"
        if not fpath.exists():
            print(f"Warning: NIC file not found for date {target_date_str}, returning None.")
            return None

        with xr.open_dataset(fpath) as ds:
            return ds.load()

    def get_ice_data_for_day(self, target_date: datetime.date) -> xr.Dataset | None:
        """Loads a pre-processed NIC NetCDF file for a given date, using a cache."""
        return self._cached_get_ice_data_for_day(target_date.strftime("%Y-%m-%d"))

    def get_water_temp_for_day(self, target_date: datetime.date) -> xr.DataArray | None:
        try:
            return self.glsea_data.sel(
                time=str(target_date),
                method="nearest",
                tolerance=pd.Timedelta(hours=12),
            )
        except (KeyError, AttributeError):
            return None

    def __getitem__(self, idx):
        H, W = self.patch_size, self.patch_size
        sample_info = self.patches[idx]
        day_t0 = sample_info["date"]
        h, w = sample_info["h"], sample_info["w"]

        # --- 1. Build Input Tensors ---
        day_t_minus_1 = day_t0 - datetime.timedelta(days=1)

        ds_t0 = self.get_ice_data_for_day(day_t0)
        ds_t_minus_1 = self.get_ice_data_for_day(day_t_minus_1)

        x_ice_t0 = (
            torch.from_numpy(ds_t0["ice_concentration"].values[h: h + H, w: w + W])
            .float()
            .unsqueeze(0)
        )
        x_ice_t_minus_1 = (
            torch.from_numpy(ds_t_minus_1["ice_concentration"].values[h: h + H, w: w + W])
            .float()
            .unsqueeze(0)
        )
        x_ice_delta = x_ice_t0 - x_ice_t_minus_1

        x_static = (
            torch.from_numpy(self.shipping_routes_mask.values[h: h + H, w: w + W])
            .float()
            .unsqueeze(0)
        )
        da_water_temp = self.get_water_temp_for_day(day_t0)
        if da_water_temp is not None:
            patch_data = da_water_temp.squeeze().values[h: h + H, w: w + W]
            water_temp_values = np.where(patch_data > 0, patch_data, 0.0)
            # Normalize Water Temp: Divide by 30.0 (Max observed ~28.8)
            water_temp_norm = water_temp_values / 30.0
            x_water_temp = (
                torch.from_numpy(np.nan_to_num(water_temp_norm)).float().unsqueeze(0)
            )
        else:
            x_water_temp = torch.zeros((1, H, W), dtype=torch.float32)

        weather_da = self.hrrr_data.sel(
            time=str(day_t_minus_1), method="nearest", tolerance=pd.Timedelta(hours=12)
        )
        weather_patch = weather_da.isel(y=slice(h, h + H), x=slice(w, w + W))
        weather_patch_normalized = np.empty((len(HRRR_VARS), H, W), dtype=np.float32)
        for i, var_name in enumerate(HRRR_VARS):
            stats = self.weather_stats[var_name]
            weather_patch_normalized[i, :, :] = (
                weather_patch[var_name].values - stats["mean"]
            ) / (stats["std"] + 1e-6)
        x_weather_tensor = torch.from_numpy(
            np.nan_to_num(weather_patch_normalized)
        ).float()

        # --- CFDD Channel ---
        # Load CFDD for the target day (T0)
        try:
            cfdd_da = self.cfdd_data.sel(time=str(day_t0), method="nearest", tolerance=pd.Timedelta(hours=12))
            cfdd_patch = cfdd_da.squeeze().values[h: h + H, w: w + W]
            
            # Normalize CFDD: Scale by a physical maximum (e.g., 3000 degree-days)
            # This preserves the positive, cumulative nature of the variable.
            # Z-score was causing negative values for early winter (0 CFDD), which ReLU suppressed.
            MAX_CFDD = 3000.0
            cfdd_patch_norm = cfdd_patch / MAX_CFDD
            
            x_cfdd = torch.from_numpy(np.nan_to_num(cfdd_patch_norm)).float().unsqueeze(0)
        except (KeyError, AttributeError, ValueError):
            x_cfdd = torch.zeros((1, H, W), dtype=torch.float32)

        # --- GEBCO Normalization ---
        # GEBCO is elevation (meters). Water is negative.
        # We want depth as positive feature 0-1.
        # Max depth ~335m. Let's divide by 400.0.
        gebco_patch = self.gebco_data.values[0, h: h + H, w: w + W]
        # Invert sign for water (make depth positive), clamp land to 0
        depth_patch = np.where(gebco_patch < 0, -gebco_patch, 0.0)
        depth_norm = depth_patch / 400.0
        x_gebco = (
            torch.from_numpy(depth_norm).float().unsqueeze(0)
        )
        
        # --- Floe Size & Edge Mask (New Channels) ---
        if "floe_size" in ds_t0:
            x_floe = torch.from_numpy(ds_t0["floe_size"].values[h: h + H, w: w + W]).float().unsqueeze(0)
        else:
            x_floe = torch.zeros((1, H, W), dtype=torch.float32)
            
        if "edge_mask" in ds_t0:
            x_edge = torch.from_numpy(ds_t0["edge_mask"].values[h: h + H, w: w + W]).float().unsqueeze(0)
        else:
            x_edge = torch.zeros((1, H, W), dtype=torch.float32)

        x = torch.cat(
            [x_ice_t0, x_ice_delta, x_weather_tensor, x_water_temp, x_static, x_gebco, x_cfdd, x_floe, x_edge],
            dim=0,
        )

        # --- 2. Build Target Tensors ---
        y_conc, y_thick = [], []
        for i in range(1, N_TIMESTEPS + 1):
            ds_t_plus_i = self.get_ice_data_for_day(day_t0 + datetime.timedelta(days=i))
            y_conc.append(
                torch.from_numpy(
                    ds_t_plus_i["ice_concentration"].values[h: h + H, w: w + W]
                ).float()
            )
            y_thick.append(
                torch.from_numpy(
                    ds_t_plus_i["ice_thickness"].values[h: h + H, w: w + W]
                ).float()
            )
        
        y = torch.stack(y_conc, dim=0)
        y_thickness = torch.stack(y_thick, dim=0)

        if config.DEBUG_MODE and idx == 0:
            config.DEBUG_DIR.mkdir(exist_ok=True)
            fig, axes = plt.subplots(4, 4, figsize=(20, 20))
            for i in range(x.shape[0]):
                ax = axes[i // 4, i % 4]
                ax.imshow(x[i].numpy())
                ax.set_title(f"Input Channel {i}")
            for i in range(y.shape[0]):
                ax = axes[(i + x.shape[0]) // 4, (i + x.shape[0]) % 4]
                ax.imshow(y[i].numpy())
                ax.set_title(f"Target Conc T+{i+1}")
            for i in range(y_thickness.shape[0]):
                ax = axes[(i + x.shape[0] + y.shape[0]) // 4, (i + x.shape[0] + y.shape[0]) % 4]
                ax.imshow(y_thickness[i].numpy())
                ax.set_title(f"Target Thick T+{i+1}")

            plt.savefig(config.DEBUG_DIR / "dataset_sample.png")
            plt.close()

        # Calculate categorical thickness targets
        # y_thickness is (T, H, W). We need (T, H, W) integers.
        # We can use apply_along_axis or just a loop since it's a small patch.
        # Vectorized approach is better.
        # But get_thickness_class is scalar.
        # Let's use np.vectorize or just simple thresholding here for speed.
        y_thick_np = y_thickness.numpy()
        y_thick_class_np = np.zeros_like(y_thick_np, dtype=np.int64)
        
        # 0: Water (<=0.001)
        # 1: New Ice (<0.10)
        # 2: Young Ice (<0.30)
        # 3: First Year Thin (<0.70)
        # 4: First Year Medium (<1.20)
        # 5: First Year Thick (>=1.20)
        
        y_thick_class_np[(y_thick_np > 0.001) & (y_thick_np < 0.10)] = 1
        y_thick_class_np[(y_thick_np >= 0.10) & (y_thick_np < 0.30)] = 2
        y_thick_class_np[(y_thick_np >= 0.30) & (y_thick_np < 0.70)] = 3
        y_thick_class_np[(y_thick_np >= 0.70) & (y_thick_np < 1.20)] = 4
        y_thick_class_np[y_thick_np >= 1.20] = 5
        
        y_thickness_class = torch.from_numpy(y_thick_class_np).long()

        return {
            "x": x,
            "y": y,
            "y_thickness": y_thickness,
            "y_thickness_class": y_thickness_class,
            "land_mask": torch.from_numpy(
                self.land_mask.values[h: h + H, w: w + W]
            ).float(),
            "shipping_mask": x_static.squeeze(0),
            "edge_mask": x_edge.squeeze(0),
            "date": str(day_t0),
        }


import re

class ConfigurableFastTensorDataset(Dataset):
    """
    A flexible dataset for loading pre-processed .pt files. It can either
    pre-load the entire dataset into RAM for speed or lazy-load from disk
    to conserve RAM.
    """
    def __init__(
        self,
        is_train: bool = True,
        val_start_date: datetime.date = None,
        shipping_routes_only: bool = False,
        pre_load: bool = False,
        min_ice_threshold: float = 0.0,
        min_thickness_threshold: float = 0.0,
        stratify_mode: bool = False,
        stratify_ratio: float = 0.5, # Target ratio of "thick" samples (0.0 to 1.0)
        stratify_threshold: float = 0.3, # Threshold to define "thick" ice
    ):
        if config.DEBUG_MODE:
            logging.debug(f"Initializing ConfigurableFastTensorDataset with is_train={is_train}, val_start_date={val_start_date}, shipping_routes_only={shipping_routes_only}, pre_load={pre_load}, min_ice={min_ice_threshold}, min_thick={min_thickness_threshold}, stratify={stratify_mode}")
        self.data_dir = config.DATA_ROOT / "processed_tensors"
        self.pre_load = pre_load
        self.is_train = is_train

        all_files = sorted(list(self.data_dir.glob("batch_*.pt")))
        if not all_files:
            raise FileNotFoundError(f"No pre-processed tensor files found in {self.data_dir}.")

        # 1. Split files into train/val sets
        if val_start_date:
            train_files, val_files = [], []
            date_pattern = re.compile(r"(\d{4}-\d{2}-\d{2})")
            for f in tqdm(all_files, desc="Splitting files by date"):
                match = date_pattern.search(f.name)
                if not match:
                    continue
                file_date = datetime.datetime.strptime(match.group(1), "%Y-%m-%d").date()
                if file_date < val_start_date:
                    train_files.append(f)
                else:
                    val_files.append(f)
            self.files = train_files if self.is_train else val_files
        else:
            num_val_batches = 2
            self.files = all_files[:-num_val_batches] if self.is_train else all_files[-num_val_batches:]

        # 2. Load data based on the pre_load flag
        self.samples = []
        self.index = []
        
        total_samples_scanned = 0
        
        # Temporary storage for stratification
        thick_candidates = [] # Stores (sample) or (file_idx, sample_idx)
        thin_candidates = []
        
        logging.info(f"Scanning files (Stratify={stratify_mode})...")
        
        for file_idx, f_path in enumerate(tqdm(self.files, desc="Scanning files")):
            samples_in_file = torch.load(f_path, weights_only=False)
            for sample_idx, sample in enumerate(samples_in_file):
                total_samples_scanned += 1
                
                # --- Base Filtering ---
                # Filter: Shipping Routes
                if shipping_routes_only and not torch.any(sample["shipping_mask"] > 0):
                    continue
                # Filter: Min Ice Threshold (Always apply this to ignore water)
                if min_ice_threshold > 0 and sample["y"].max() < min_ice_threshold:
                    continue
                
                # --- Stratification or Thresholding ---
                max_thick = sample.get("y_thickness", torch.zeros_like(sample["y"])).max()
                
                if stratify_mode:
                    item_to_store = sample if self.pre_load else (file_idx, sample_idx)
                    if max_thick >= stratify_threshold:
                        thick_candidates.append(item_to_store)
                    else:
                        thin_candidates.append(item_to_store)
                else:
                    # Standard Thresholding
                    if min_thickness_threshold > 0 and max_thick < min_thickness_threshold:
                        continue
                        
                    if self.pre_load:
                        self.samples.append(sample)
                    else:
                        self.index.append((file_idx, sample_idx))

        # 3. Apply Stratification Logic
        if stratify_mode:
            num_thick = len(thick_candidates)
            num_thin = len(thin_candidates)
            logging.info(f"Stratification Candidates: {num_thick} Thick (>{stratify_threshold}m), {num_thin} Thin")
            
            if num_thick == 0:
                logging.warning("No thick samples found! Falling back to all thin samples.")
                final_items = thin_candidates
            elif num_thin == 0:
                logging.warning("No thin samples found! Using only thick samples.")
                final_items = thick_candidates
            else:
                # Calculate how many thin samples to keep to match the ratio
                # target_thick_ratio = num_thick / (num_thick + num_thin_to_keep)
                # num_thin_to_keep = num_thick * (1 - ratio) / ratio
                
                if stratify_ratio >= 1.0:
                    num_thin_to_keep = 0
                else:
                    num_thin_to_keep = int(num_thick * (1.0 - stratify_ratio) / stratify_ratio)
                
                # Clamp to available thin samples
                num_thin_to_keep = min(num_thin_to_keep, num_thin)
                
                # Randomly select thin samples
                import random
                random.shuffle(thin_candidates)
                selected_thin = thin_candidates[:num_thin_to_keep]
                
                final_items = thick_candidates + selected_thin
                random.shuffle(final_items)
                
                logging.info(f"Stratified Selection: Kept {len(thick_candidates)} Thick + {len(selected_thin)} Thin = {len(final_items)} Total. (Target Ratio: {stratify_ratio})")

            if self.pre_load:
                self.samples = final_items
            else:
                self.index = final_items
        else:
            logging.info(f"Dataset (Train={self.is_train}, Pre-load={self.pre_load}): {len(self.samples) if self.pre_load else len(self.index)} samples kept out of {total_samples_scanned} scanned.")

        num_samples = len(self.samples) if self.pre_load else len(self.index)
        if num_samples == 0:
            logging.warning("No samples found for this dataset configuration.")

    @lru_cache(maxsize=1)
    def _load_file(self, file_idx: int):
        return torch.load(self.files[file_idx], weights_only=False)

    def __len__(self):
        return len(self.samples) if self.pre_load else len(self.index)

    def __getitem__(self, idx):
        # DEBUG: Trace getitem
        # if idx % 100 == 0:
        #     print(f"Loading sample {idx}")
            
        if self.pre_load:
            sample = self.samples[idx]
        else:
            file_idx, sample_idx_in_file = self.index[idx]
            samples_in_file = self._load_file(file_idx)
            sample = samples_in_file[sample_idx_in_file]


            
        return sample


class FileAwareSampler(Sampler):
    """
    A PyTorch Sampler that shuffles data by first shuffling files (batches)
    and then shuffling samples within each file.
    """
    def __init__(self, dataset: 'ConfigurableFastTensorDataset'):
        """
        Args:
            dataset (Dataset): The dataset to sample from. Must have an `index`
                               attribute as described above.
        """
        super().__init__(dataset)
        if dataset.pre_load:
            raise ValueError("FileAwareSampler is not needed when pre_load is True.")

        self.dataset = dataset
        self.indices_by_file = defaultdict(list)
        for i, (file_idx, _) in enumerate(self.dataset.index):
            self.indices_by_file[file_idx].append(i)

        self.num_samples = len(self.dataset.index)
        if config.DEBUG_MODE:
            logging.debug(f"FileAwareSampler indexed {len(self.indices_by_file)} files and {self.num_samples} total samples.")

    def __iter__(self):
        # Get a list of file indices and shuffle them
        file_indices = list(self.indices_by_file.keys())
        np.random.shuffle(file_indices)

        # Iterate through shuffled files and then shuffled samples within each file
        for file_idx in file_indices:
            sample_indices_for_file = self.indices_by_file[file_idx]
            np.random.shuffle(sample_indices_for_file)
            for sample_idx in sample_indices_for_file:
                yield sample_idx

    def __len__(self):
        return self.num_samples