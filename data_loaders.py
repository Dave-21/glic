import torch
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
import numpy as np
import datetime
from pathlib import Path
import glob
from typing import List, Dict
import pandas as pd
import re
import s3fs
import zarr
import geopandas as gpd
from rasterio import features
from rasterio.transform import from_bounds
import warnings
import metpy
import cartopy.crs as ccrs
from tqdm import tqdm
import os
import bathyreq
import traceback
import config
import utilities


# --- Cache Variables ---
_hrrr_day_cache = {}
_nic_ice_data_cache = None
_hrrr_grid_cache = None
_glsea_ice_data_cache = None
_gebco_data_cache = None

# --- HRRR Projection (from HRRR Guide) ---
HRRR_PROJECTION = ccrs.LambertConformal(
    central_longitude=262.5, 
    central_latitude=38.5, 
    standard_parallels=(38.5, 38.5),
    globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229)
)

def load_hrrr_data_for_day(target_date: datetime.date, master_grid: xr.DataArray) -> xr.Dataset | None:
    """
    Loads HRRR data for a single day from the S3 Zarr archive and reprojects it.
    This version is robust to format changes over time by loading coordinates
    from *within* the variable groups, not the root.
    """
    if target_date in _hrrr_day_cache:
        return _hrrr_day_cache[target_date].copy()

    zarr_url = f"s3://hrrrzarr/sfc/{target_date.strftime('%Y%m%d')}/{target_date.strftime('%Y%m%d')}_00z_anl.zarr"
    
    try:
        #print(f"--- Loading HRRR data for {target_date} from S3 ---")
        
        fs = s3fs.S3FileSystem(anon=True)
        mapper = s3fs.S3Map(root=zarr_url, s3=fs, check=False)
        
        # 1. Open the main Zarr store
        store = zarr.open_consolidated(mapper, mode="r")

        # 2. Load each variable manually.
        loaded_vars = []
        ds_grid = None  # This will store our coordinate system for the day

        for internal_name, group_path in config.HRRR_VAR_MAP.items():
            try:
                var_group = store[group_path]

                # --- LOAD COORDINATES FROM GROUP ---
                if ds_grid is None:
                    #print(f"  > Initializing coordinate system from group: {group_path}")
                    
                    if 'projection_y_coordinate' in var_group.array_keys():
                        #print("    > Found 'projection_y_coordinate', renaming...")
                        y_coords = var_group['projection_y_coordinate'][:]
                        x_coords = var_group['projection_x_coordinate'][:]
                        temp_ds = xr.Dataset(coords={'projection_y_coordinate': y_coords, 'projection_x_coordinate': x_coords})
                        ds_grid_renamed = temp_ds.rename({'projection_y_coordinate': 'y', 'projection_x_coordinate': 'x'})
                    elif 'y' in var_group.array_keys():
                        #print("    > Found 'y'/'x' coordinates directly.")
                        y_coords = var_group['y'][:]
                        x_coords = var_group['x'][:]
                        ds_grid_renamed = xr.Dataset(coords={'y': y_coords, 'x': x_coords})
                    else:
                        raise KeyError(f"Could not find 'y' or 'projection_y_coordinate' in group {group_path}")

                    ds_grid_with_crs = ds_grid_renamed.metpy.assign_crs(
                        grid_mapping_name='lambert_conformal_conic',
                        **HRRR_PROJECTION.proj4_params
                    )
                    
                    # Set CRS, then set dims, then infer transform. All inplace.
                    ds_grid_with_crs.rio.write_crs(HRRR_PROJECTION.proj4_init, inplace=True)
                    ds_grid_with_crs.rio.set_spatial_dims("x", "y", inplace=True)
                    
                    # Manually calculate the transform instead of inferring
                    #print("  > Manually calculating HRRR grid transform...")
                    transform = from_bounds(
                        west=ds_grid_with_crs['x'].values.min(),
                        south=ds_grid_with_crs['y'].values.min(),
                        east=ds_grid_with_crs['x'].values.max(),
                        north=ds_grid_with_crs['y'].values.max(),
                        width=len(ds_grid_with_crs['x']),
                        height=len(ds_grid_with_crs['y'])
                    )
                    ds_grid_with_crs.rio.write_transform(transform, inplace=True)
                    
                    ds_grid = ds_grid_with_crs
                    ds_grid.load()
                    #print(f"  > Coordinate system loaded. CRS: {ds_grid.rio.crs}")

                # --- ROBUST ZARR ARRAY FINDER ---
                short_name = group_path.split('/')[-1] 
                base_name = group_path.split('/')[0]   
                var_array = None
                array_path_for_logging = "???"

                if short_name in var_group.array_keys():
                    array_path_for_logging = f"{group_path}/{short_name}"
                    #print(f"  > Found flat-style array: {array_path_for_logging}")
                    var_array = var_group[short_name]
                    
                elif base_name in var_group.group_keys():
                    sub_group = var_group[base_name]
                    if short_name in sub_group.array_keys():
                        array_path_for_logging = f"{group_path}/{base_name}/{short_name}"
                        #print(f"  > Found nested-style array: {array_path_for_logging}")
                        var_array = sub_group[short_name]
                
                elif group_path in var_group.array_keys():
                     array_path_for_logging = f"{group_path}/{group_path}"
                     #print(f"  > Found original-hack-style array: {array_path_for_logging}")
                     var_array = var_group[group_path]
                
                if var_array is None:
                    raise KeyError(f"Could not find data array in known locations for {group_path}")

                #print(f"  > Array found. Shape: {var_array.shape}, Dims: {var_array.ndim}")
                
                if var_array.ndim == 3:
                    #print("  > Slicing 3D array at index 12.")
                    var_data = var_array[12, :, :].astype(np.float32)
                elif var_array.ndim == 2:
                    #print("  > Using 2D array as-is.")
                    var_data = var_array[:].astype(np.float32)
                else:
                    raise ValueError(f"Unexpected array shape for {array_path_for_logging}")
                
                coords = {'y': ds_grid['y'], 'x': ds_grid['x']}
                dims = ['y', 'x']
                
                loaded_vars.append(xr.DataArray(
                    var_data,
                    coords=coords,
                    dims=dims,
                    name=internal_name
                ))
                
            except Exception as e:
                print(f"!!! ERROR loading path '{group_path}' (from base '{internal_name}') for {target_date}: {e}")
                traceback.print_exc()
                raise e 
        
        if not loaded_vars or ds_grid is None:
            raise ValueError("No HRRR variables could be loaded or grid was not created.")

        # 3. Create an xarray.Dataset from the loaded variables.
        ds_at_12utc = xr.merge(loaded_vars)
        
        # 4. Merge with the coordinate/grid dataset
        ds_with_coords = xr.merge([ds_at_12utc, ds_grid])

        # Because `ds_grid.rio.crs` is now valid
        ds_with_coords.rio.write_crs(ds_grid.rio.crs, inplace=True)

        # 5. Reproject to Master Grid.
        
        #print(f"Reprojecting HRRR for {target_date} to master grid...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            
            # We must iterate *only* over the weather variables from config,
            # not all data_vars (which includes 'metpy_crs').
            
            reprojected_vars = []
            # Use the list of var names from config to ensure we only get
            # the 4 variables we care about.
            for var_name in config.HRRR_VARS: 
                
                # Select the DataArray from the main dataset
                data_arr = ds_with_coords[var_name]
                
                # Default to bilinear for temp, wind, etc.
                resampling_method = Resampling.bilinear
                
                if var_name == 'precip_surface':
                    # Precipitation is a rate, not a continuous field.
                    # Bilinear will create false negative values.
                    # 'Nearest' is safer and preserves 0s.
                    resampling_method = Resampling.nearest
                    #print(f"  > Using 'nearest' resampling for {var_name}")
                
                # Re-project this single DataArray
                var_reprojected = data_arr.rio.reproject_match(
                    master_grid,
                    resampling=resampling_method
                )
                
                if var_name == 'precip_surface':
                    # Clip all negative values (artifacts from reprojection) to 0.0
                    var_reprojected = var_reprojected.clip(min=0.0)
                    #print(f"  > Clipping negative precipitation artifacts.")

                reprojected_vars.append(var_reprojected)

            # Merge the reprojected variables back into one dataset
            #ds_reprojected = xr.merge(reprojected_vars)
            ds_reprojected = xr.merge(reprojected_vars, compat='override')

        ds_reprojected["time"] = pd.to_datetime(target_date) # Add date for caching
        
        # 6. Cache and return.
        ds_reprojected.load()
        _hrrr_day_cache[target_date] = ds_reprojected
        print(f"--- Successfully loaded HRRR for {target_date} ---")
        return ds_reprojected.copy()

    except Exception as e:
        print(f"!!! WARNING: Could not load HRRR data for {target_date}")
        print(f"    URL: {zarr_url}")
        print(f"    Error: {e}")
        return None

def load_glsea_ice_data() -> xr.DataArray:
    """
    Loads and processes GLSEA ice data from NetCDF files.
    Combines 2018 and 2019 data, applies the rotation fix, and
    *manually assigns real-world Lat/Lon coordinates* to georeference it.
    """
    global _glsea_ice_data_cache

    # 1. Memory Cache Check
    if _glsea_ice_data_cache is not None:
        print("GLSEA ice data retrieved from memory cache.")
        return _glsea_ice_data_cache.copy()

    # 2. Disk Cache Check
    if config.TRAIN_GLSEA_ICE_CACHE_FILE.exists():
        print(f"Loading GLSEA ice data from disk cache: {config.TRAIN_GLSEA_ICE_CACHE_FILE}")
        try:
            with xr.open_dataset(config.TRAIN_GLSEA_ICE_CACHE_FILE, engine="netcdf4") as ds:
                _glsea_ice_data_cache = ds['glsea_ice_temp'].load()
                print("GLSEA ice data load complete from disk cache.")
                return _glsea_ice_data_cache
        except Exception as e:
            print(f"Error loading GLSEA ice cache file ({e}), re-running expensive computation.")

    print("Loading and processing GLSEA ice data (this is a slow, one-time process)...")

    all_glsea_ice_das = []

    # We are defining the grid's extent in Lat/Lon (EPSG:4326)
    # Based on the rotation, 'ny' (1024) is 'x' (Longitude)
    # and 'nx' (1024) is 'y' (Latitude)
    N_X = 1024 # ny
    N_Y = 1024 # nx

    # Longitude (x-axis)
    lons = np.linspace(-92.5, -75.5, N_X) # West to East
    # Latitude (y-axis)
    #lats = np.linspace(49.5, 41.0, N_Y)  # North to South (descending)
    lats = np.linspace(41.0, 49.5, N_Y)  # South to North (ascending)

    for file_path in config.TRAIN_GLSEA_ICE_NC_FILES:
        if not file_path.exists():
            print(f"!!! WARNING: GLSEA ice file not found: {file_path}, skipping.")
            continue

        try:
            with xr.open_dataset(file_path) as ds:
                glsea_da = ds['temp'] # shape (time, nx, ny)

                # 1. Rename dims: 'nx' -> 'y', 'ny' -> 'x'
                #    This makes the dims ('time', 'y', 'x')
                glsea_da = glsea_da.rename({'nx': 'y', 'ny': 'x'})

                # 2. Assign the Lat/Lon coordinates we defined
                glsea_da = glsea_da.assign_coords({'x': lons, 'y': lats})

                # --- SET SPATIAL DIMS & CRS ---
                # 3. Tell rioxarray what the dims and CRS are
                glsea_da = glsea_da.rio.set_spatial_dims(x_dim='x', y_dim='y')
                glsea_da.rio.write_crs(config.MASTER_GRID_CRS, inplace=True)

                all_glsea_ice_das.append(glsea_da)

        except Exception as e:
            print(f"Error processing GLSEA ice file {file_path}: {e}")
            traceback.print_exc()

    if not all_glsea_ice_das:
        raise ValueError("Could not create GLSEA ice dataset from provided files.")

    full_glsea_ice_dataset = xr.concat(all_glsea_ice_das, dim="time")
    full_glsea_ice_dataset = full_glsea_ice_dataset.sortby("time")

    # We don't need to reproject, we just need to load.
    full_glsea_ice_dataset.load()

    print(f"Caching GLSEA ice data to: {config.TRAIN_GLSEA_ICE_CACHE_FILE}")
    ds_to_save = full_glsea_ice_dataset.to_dataset(name='glsea_ice_temp')
    config.TRAIN_GLSEA_ICE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    ds_to_save.to_netcdf(config.TRAIN_GLSEA_ICE_CACHE_FILE, engine="h5netcdf")

    print("--- DEBUG: Master GLSEA Ice Data Stats ---")
    print(f"  Shape: {full_glsea_ice_dataset.shape}")
    print(f"  Min:   {full_glsea_ice_dataset.min().item():.4f}")
    print(f"  Max:   {full_glsea_ice_dataset.max().item():.4f}")
    print(f"  Mean:  {full_glsea_ice_dataset.mean().item():.4f}")
    print(f"  NaNs:  {np.isnan(full_glsea_ice_dataset.values).sum()} (Land pixels)")
    print("-----------------------------------------------------")

    _glsea_ice_data_cache = full_glsea_ice_dataset
    print("GLSEA ice data cached.")
    return _glsea_ice_data_cache

def get_glsea_ice_data() -> xr.DataArray:
    """
    Wrapper function to get the GLSEA ice data, using the cache if available.
    """
    global _glsea_ice_data_cache
    if _glsea_ice_data_cache is None:
        _glsea_ice_data_cache = load_glsea_ice_data()
    return _glsea_ice_data_cache.copy()

def load_gebco_data(master_grid: xr.DataArray) -> xr.DataArray:
    """
    Loads and processes GEBCO bathymetry data using bathyreq.
    """
    global _gebco_data_cache

    # 1. Memory Cache Check
    if _gebco_data_cache is not None:
        print("GEBCO data retrieved from memory cache.")
        return _gebco_data_cache.copy()

    # 2. Disk Cache Check
    if config.GEBCO_FILE.exists():
        print(f"Loading GEBCO data from disk cache: {config.GEBCO_FILE}")
        try:
            with rioxarray.open_rasterio(config.GEBCO_FILE) as da:
                _gebco_data_cache = da.load()
                print("GEBCO data load complete from disk cache.")
                return _gebco_data_cache
        except Exception as e:
            print(f"Error loading GEBCO cache file ({e}), re-running expensive computation.")

    print("Loading and processing GEBCO data (this is a slow, one-time process)...")
    
    try:
        # Download the data using bathyreq
        req = bathyreq.BathyRequest()
        data, lon, lat = req.get_area(
            longitude=[-93.1, -74.9],
            latitude=[40.9, 49.6],
            resolution='20m'
        )
        
        # Create an xarray DataArray from the downloaded data
        da = xr.DataArray(
            data,
            coords={'y': lat, 'x': lon},
            dims=['y', 'x'],
        )
        da.rio.write_crs("EPSG:4326", inplace=True)

        print("Reprojecting GEBCO data to master grid...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            gebco_reprojected = da.rio.reproject_match(
                master_grid,
                resampling=Resampling.bilinear,
            )
        
        # Save to cache
        gebco_reprojected.rio.to_raster(config.GEBCO_FILE)
        _gebco_data_cache = gebco_reprojected.load()
        return _gebco_data_cache

    except Exception as e:
        print(f"Error processing GEBCO data: {e}")
        raise e

def get_gebco_data(master_grid: xr.DataArray) -> xr.DataArray:
    """
    Wrapper function to get the GEBCO data, using the cache if available.
    """
    global _gebco_data_cache
    if _gebco_data_cache is None:
        _gebco_data_cache = load_gebco_data(master_grid)
    return _gebco_data_cache.copy()

"""
Test Data Loaders
"""
def load_test_ice_raw() -> xr.DataArray | None:
    """
    Loads the test set initial condition for ice concentration (RAW).
    Converts 0-100 to 0-1 for consistency, but keeps land as NaN for now.
    """
    try:
        with xr.open_dataset(config.TEST_ICE_NC, engine="netcdf4", decode_times=False) as ds:
            data_da = ds['ice_cover'].load().squeeze() # Values 0-100, NaN for land

            # --- 1. Rename dims to 'x' and 'y' ---
            rename_dict = {}
            if 'lon' in data_da.coords: rename_dict['lon'] = 'x'
            if 'lat' in data_da.coords: rename_dict['lat'] = 'y'
            data_da = data_da.rename(rename_dict)
            
            # --- 2. Steal coordinates from the master grid ---
            print("...Assigning master grid coordinates to test ice data...")
            master_grid = utilities.get_master_grid()
            
            # Flip data to match ascending 'y' coords
            data_da.values = np.flipud(data_da.values)

            data_da = data_da.assign_coords({
                'x': master_grid.coords['x'],
                'y': master_grid.coords['y']
            })
            
            # --- 3. Set spatial dims & CRS ---
            data_da = data_da.rio.set_spatial_dims(x_dim='x', y_dim='y')
            data_da.rio.write_crs(config.MASTER_GRID_CRS, inplace=True)
            
            # --- 4. Process like in training ---
            # Convert 0-100 to 0-1
            data_da = data_da / 100.0
            # Set land (NaN) to 0.0, as done in training
            data_da_clean = data_da.fillna(0.0) 
            
            return data_da_clean

    except Exception as e:
        print(f"Error loading test ice concentration data: {e}")
        return None

def load_test_water_temp_raw() -> xr.DataArray | None:
    """
    Loads the test set initial condition for water temp (RAW).
    """
    try:
        with xr.open_dataset(config.TEST_GLSEA_NC, engine="netcdf4", decode_times=False) as ds:
            data_da = ds['sst'].load().squeeze() # Values in C, NaN for land/ice

            # --- 1. Rename dims to 'x' and 'y' ---
            rename_dict = {}
            if 'lon' in data_da.coords: rename_dict['lon'] = 'x'
            if 'lat' in data_da.coords: rename_dict['lat'] = 'y'
            data_da = data_da.rename(rename_dict)

            # --- 2. Steal coordinates from the master grid ---
            print("...Assigning master grid coordinates to test water temp data...")
            master_grid = utilities.get_master_grid()
            
            # Flip data to match ascending 'y' coords
            data_da.values = np.flipud(data_da.values)

            data_da = data_da.assign_coords({
                'x': master_grid.coords['x'],
                'y': master_grid.coords['y']
            })
            
            # --- 3. Set spatial dims & CRS ---
            data_da = data_da.rio.set_spatial_dims(x_dim='x', y_dim='y')
            data_da.rio.write_crs(config.MASTER_GRID_CRS, inplace=True)

            # --- 4. Process like in training ---
            # Set land/ice (NaN) to 0.0
            data_da_clean = data_da.fillna(0.0)
            
            return data_da_clean

    except Exception as e:
        print(f"Error loading test water temp data: {e}")
        return None

def load_test_hrrr_forecast_raw() -> xr.Dataset | None:
    """
    Loads the full 4-day (96hr) HRRR forecast (RAW).
    This dataset MUST be reprojected before use.
    """
    try:
        with xr.open_dataset(config.TEST_HRRR_NC, engine="netcdf4", decode_times=False) as ds:
            ds_load = ds.load()

            rename_dict = {}
            if 'lon' in ds_load.coords: rename_dict['lon'] = 'x'
            if 'lat' in ds_load.coords: rename_dict['lat'] = 'y'
            ds_load = ds_load.rename(rename_dict)
            
            # --- 0-360 longitude ---
            if 'x' in ds_load.coords and ds_load['x'].max() > 180:
                print("...Converting HRRR longitude from 0-360 to -180-180...")
                ds_load.coords['x'] = (((ds_load.coords['x'] + 180) % 360) - 180)
                ds_load = ds_load.sortby('x')

            ds_load = ds_load.rio.set_spatial_dims(x_dim='x', y_dim='y')
            
            if ds_load.rio.crs is None:
                ds_load.rio.write_crs(config.MASTER_GRID_CRS, inplace=True)
                
            return ds_load

    except Exception as e:
        print(f"Error loading test HRRR data: {e}")
        return None

def load_test_initial_conditions(master_grid: xr.DataArray):
    """
    Loads and returns the initial conditions for the test forecast.
    """
    ice_raw = load_test_ice_raw()
    water_temp_raw = load_test_water_temp_raw()

    # ice_class is not provided, so we derive it from ice concentration
    # This is a simple assumption, but it's better than nothing.
    ice_class = (ice_raw > 0.15).astype(np.float32)

    return ice_raw.values, water_temp_raw.values, ice_class.values

def load_test_ground_truth(dates: list) -> dict:
    """
    Loads the ground truth ice concentration for the given dates.
    """
    import data_loaders
    import pandas as pd

    master_ice_and_temp_data = data_loaders.get_glsea_ice_data()
    ground_truth_data = {}

    for date in dates:
        date_pd = pd.to_datetime(date)
        da = master_ice_and_temp_data.sel(time=date_pd, method="nearest")
        time_diff = abs((da.time.values - date_pd.to_numpy()) / np.timedelta64(1, 'h'))

        if time_diff <= 12:
            da_squeezed = da.squeeze()
            ice_conc = xr.where(da_squeezed < 0, -da_squeezed, 0.0)
            ground_truth_data[date] = ice_conc
        else:
            ground_truth_data[date] = None

    return ground_truth_data