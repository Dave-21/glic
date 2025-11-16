import traceback
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
import numpy as np
import datetime

import config

_master_grid_def = None
_land_mask = None

def get_master_grid_definition() -> xr.DataArray:
    """
    Loads (or creates) the master grid definition from the GLSEA template file.
    This version is based on your 'utils_fix.py' and handles 2D coordinates.
    """
    global _master_grid_def
    if _master_grid_def is not None:
        return _master_grid_def.copy()

    print(f"Loading master grid template from: {config.MASTER_GRID_TEMPLATE_FILE}")
    
    try:
        with xr.open_dataset(config.MASTER_GRID_TEMPLATE_FILE, engine="netcdf4", decode_times=False) as ds:
            var_name = 'temp'
            if var_name not in ds.variables: var_name = 'sst'
            if var_name not in ds.variables:
                 raise ValueError(f"Could not find 'temp' or 'sst' in {config.MASTER_GRID_TEMPLATE_FILE}.")
                 
            master_da = ds[var_name].isel(time=0).load().squeeze()
            
            print("[DEBUG] Forcing 1D coordinates on master grid...")
            
            # Use 'lat' and 'lon' as they are the standard coord names
            lats_2d = ds['lat']
            lons_2d = ds['lon']
            
            if lons_2d.max() > 180:
                print("...Converting master grid longitude from 0-360 to -180-180...")
                lons_2d = (((lons_2d + 180) % 360) - 180)

            y_dim_name, x_dim_name = master_da.dims[-2], master_da.dims[-1]
            y_size = len(master_da[y_dim_name])
            x_size = len(master_da[x_dim_name])

            new_y = np.linspace(lats_2d.min().item(), lats_2d.max().item(), y_size)
            new_x = np.linspace(lons_2d.min().item(), lons_2d.max().item(), x_size)
            
            # --- FIX: Flip data to match ascending coordinates ---
            # master_da.values[0] is North, new_y[0] is South.
            # We must flip the data array to match the coordinates.
            flipped_values = np.flipud(master_da.values)
            
            master_da = xr.DataArray(
                flipped_values,  # Use the flipped data
                coords={'y': new_y, 'x': new_x},
                dims=['y', 'x']
            )
            print("[DEBUG] 1D coordinates assigned (with vertical flip).")

            if master_da.rio.crs is None:
                print(f"Master grid has no CRS. Writing default: {config.MASTER_GRID_CRS}")
                master_da.rio.write_crs(config.MASTER_GRID_CRS, inplace=True)
            
            master_da = master_da.rio.set_spatial_dims(x_dim='x', y_dim='y')
            
            _master_grid_def = master_da
            print(f"Master grid loaded. Shape: {_master_grid_def.shape}, CRS: {_master_grid_def.rio.crs}")
            return _master_grid_def.copy()

    except Exception as e:
        print(f"FATAL Error in get_master_grid_definition: {e}")
        traceback.print_exc()
        _master_grid_def = None
        return None

def get_land_mask() -> xr.DataArray | None:
    """
    Loads the 'ice asc' file and assigns the MASTER GRID coordinates to it
    (per the README, they are on the "same grid").
    
    This function NO LONGER reprojects.
    """
    global _land_mask
    if _land_mask is not None:
        return _land_mask.copy()

    print("--- STARTING get_land_mask (v6 - README FIX) ---")
    
    # 1. Get the master grid to steal its coordinates
    master_grid = get_master_grid_definition()
    
    # 2. Load the raw .asc data (any date will do for a mask)
    print("[v6 DEBUG] 1. Loading raw 'ice asc' data (header is ignored)...")
    
    # We must import here to avoid circular dependencies
    import data_loaders
    
    # Try to load the first day from the training period
    # (The .ct file from the test data isn't in the training dir)
    raw_asc_data = data_loaders.load_ice_asc(datetime.date(2019, 1, 11))
    
    if raw_asc_data is None:
        print("FATAL: Could not load 'g20190111.ct' to generate land mask.")
        return None

    # 3. Create the mask (0=water, 1=land)
    # The .asc file uses -1 for land, >= 0 for water.
    print("[v6 DEBUG] 2. Creating mask from raw data...")
    land_mask_data = (raw_asc_data == config.ICE_ASC_NODATA_VAL).astype(int)

    # 4. CRITICAL FIX: Assign the master grid's coordinates and CRS
    print("[v6 DEBUG] 3. Assigning master grid coordinates (NO REPROJECTION)...")
    
    # --- FIX: Flip data to match ascending coordinates ---
    # The .asc data is also top-to-bottom, so we flip it
    # to match the now-correct master_grid.
    flipped_land_mask_values = np.flipud(land_mask_data.values)
    
    land_mask_da = xr.DataArray(
        flipped_land_mask_values,  # Use the flipped data
        coords=master_grid.coords, # Steal coords from master grid
        dims=master_grid.dims,     # Steal dims from master grid
        attrs=master_grid.attrs    # Steal CRS and attrs from master grid
    )
    
    # 5. Set spatial dims just to be safe
    land_mask_da = land_mask_da.rio.set_spatial_dims(x_dim='x', y_dim='y')
    if land_mask_da.rio.crs is None:
        land_mask_da.rio.write_crs(config.MASTER_GRID_CRS, inplace=True)

    print(f"Land mask generated. Shape: {land_mask_da.shape}, CRS: {land_mask_da.rio.crs}")
    print("--- ENDING get_land_mask (SUCCESS) ---")
    
    _land_mask = land_mask_da.copy()
    return _land_mask

def reproject_to_master(source_da: xr.DataArray, resampling_method) -> xr.DataArray:
    """Reprojects a DataArray to match the master grid."""
    master_grid_def = get_master_grid_definition()
    
    if master_grid_def is None:
        raise Exception("Could not reproject, master grid is not loaded.")
        
    if source_da.rio.crs is None:
        raise ValueError("Source DataArray must have a CRS defined. Use .rio.write_crs()")
    
    # Check if already on the same grid
    if (source_da.rio.crs == master_grid_def.rio.crs and 
        source_da.shape == master_grid_def.shape):
        print("...Skipping reprojection; grid already matches.")
        return source_da.assign_coords({
            'x': master_grid_def.coords['x'],
            'y': master_grid_def.coords['y']
        })

    print(f"Reprojecting from CRS {source_da.rio.crs} and shape {source_da.shape}...")
    
    return source_da.rio.reproject_match(
        master_grid_def,
        resampling=resampling_method
    )

def print_da_stats(da, name: str):
    """Prints debugging stats for an xarray DataArray or Dataset."""
    print("\n" + "="*30)
    print(f"--- DEBUG STATS FOR: {name} ---")
    
    if da is None:
        print("  !!! DATA IS NONE !!!")
        print("="*30 + "\n")
        return

    print(f"  Type: {type(da)}")
    
    if isinstance(da, xr.Dataset):
        print(f"  Dimensions: {da.sizes}")
        print(f"  Data Variables: {list(da.data_vars)}")
        print(f"  Coordinates: {list(da.coords)}")
        print("="*30 + "\n")
        return

    print(f"  Shape: {da.shape}")
    
    try:
        if isinstance(da, xr.DataArray) or isinstance(da, xr.Dataset):
            values = da.values
        else:
            values = da # It's already a numpy array

        min_val = np.nanmin(values)
        max_val = np.nanmax(values)
        mean_val = np.nanmean(values)
        nan_count = np.isnan(values).sum()
        total_count = values.size
        
        print(f"  Min:   {min_val:.4f}")
        print(f"  Max:   {max_val:.4f}")
        print(f"  Mean:  {mean_val:.4f}")
        print(f"  NaNs:  {nan_count} / {total_count} ({nan_count/total_count:.1%})")

        if name.lower().startswith('land mask'):
            water_pixels = np.count_nonzero(values == 0)
            land_pixels = np.count_nonzero(values == 1)
            print(f"  Water (0s): {water_pixels}")
            print(f"  Land (1s):  {land_pixels}")
            if water_pixels == 0:
                print("  !!! CRITICAL: No water pixels found in land mask!")
                
        elif 'initial' in name.lower() or 'test' in name.lower():
             if (total_count > 0 and (nan_count / total_count) > 0.99):
                 print(f"  !!! CRITICAL: Data is almost all NaN! Reprojection likely failed.")

        elif 'forecast' in name.lower() or 'submission' in name.lower():
            land_pixels = np.count_nonzero(values == -1)
            water_pixels = np.count_nonzero(values >= 0)
            print(f"  Land (-1s): {land_pixels}")
            print(f"  Water (>=0): {water_pixels}")
            if water_pixels == 0:
                print("  !!! CRITICAL: No water data in final forecast!")

    except Exception as e:
        print(f"  !!! Could not compute stats: {e} !!!")
        
    print("="*30 + "\n")