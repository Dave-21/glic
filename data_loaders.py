import xarray as xr
import rioxarray
from rasterio.enums import Resampling
import numpy as np
import datetime
from pathlib import Path
import glob

import config
# We remove 'import utils' from here to be safe,
# as this file should not depend on utils.

# 0=Land/Unknown (default)
ICECON_MAP = {
    1: 1,  # Calm Water
    21: 2, # New Lake Ice
    12: 3, # Pancake Ice
    27: 4, # Consolidated Floes
    14: 5  # Brash Ice
}

def load_all_icecon(master_grid: xr.DataArray) -> xr.Dataset:
    """
    Loads all ICECON .nc files, remaps their values, reprojects them,
    and combines them into a single time-series xarray Dataset.
    """
    print("Loading all ICECON data into memory...")
    search_path = str(config.TRAIN_ICECON_NC_DIR / "*.nc")
    all_files = sorted(glob.glob(search_path))
    
    if not all_files:
        raise IOError(f"No ICECON .nc files found in {config.TRAIN_ICECON_NC_DIR}")

    # Vectorized mapping function
    def map_func(val):
        return ICECON_MAP.get(val, 0) # Default to 0
    
    map_vectorized = np.vectorize(map_func)
    
    #daily_arrays = []
    daily_data_groups = {}
    
    for f in all_files:
        try:
            # 1. Extract date from filename (e.g., ..._2019_01_11_...)
            fname = Path(f).stem
            parts = fname.split('_')
            date_str = f"{parts[2]}-{parts[3]}-{parts[4]}" # YYYY-MM-DD
            date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()

            # 2. Load and process the file
            with xr.open_dataset(f, decode_cf=False, engine="netcdf4") as ds:
                da = ds['iceclass'].load().squeeze() # This is the variable with codes [cite: 12]

                # Rename dims if necessary
                rename_dict = {}
                if 'longitude' in da.coords: rename_dict['longitude'] = 'x'
                if 'latitude' in da.coords: rename_dict['latitude'] = 'y'
                da = da.rename(rename_dict)
                
                #da = da.rio.set_spatial_dims(x_dim='x', y_dim='y')
                if da.rio.crs is None:
                    # Assume master grid CRS if missing
                    da.rio.write_crs(config.MASTER_GRID_CRS, inplace=True)
                
                # 3. Reproject (use nearest neighbor for categorical data)
                da_reproj = da.rio.reproject_match(
                    master_grid,
                    resampling=Resampling.nearest 
                )
                
                # 4. Apply mapping to our 0-5 scheme
                mapped_values = map_vectorized(da_reproj.values)
                
                mapped_da = xr.DataArray(
                    mapped_values,
                    coords=da_reproj.coords,
                    dims=da_reproj.dims,
                    attrs=da_reproj.attrs
                )
                
                # Group the processed array by its date. We will merge swaths later.
                if date_obj not in daily_data_groups:
                    daily_data_groups[date_obj] = []
                daily_data_groups[date_obj].append(mapped_da)

        except Exception as e:
            print(f"Warning: Could not load or process ICECON file {f}: {e}")
    
    if not daily_data_groups:
        raise IOError("Failed to load any valid ICECON data.")

    print(f"Merging {len(all_files)} files into {len(daily_data_groups)} unique days...")
    final_daily_arrays = []
    
    # Sort by date to ensure the final series is in order
    for date_obj, arrays_for_this_date in sorted(daily_data_groups.items()):
        
        # Use combine_first to merge multiple swaths (files) from the same day
        # This "patches" NaNs from the first array with values from the next
        combined_da = arrays_for_this_date[0]
        if len(arrays_for_this_date) > 1:
            for i in range(1, len(arrays_for_this_date)):
                combined_da = combined_da.combine_first(arrays_for_this_date[i])
        
        # NOW assign the single time coordinate to the merged daily map
        combined_da = combined_da.assign_coords(time=date_obj)
        final_daily_arrays.append(combined_da)

    # 5. Combine into a single xarray object (use the new list of unique daily maps)
    combined_ds = xr.concat(final_daily_arrays, dim='time').to_dataset(name='ice_class')
    print(f"Loaded and remapped {len(combined_ds.time)} ICECON time steps.")
    return combined_ds.load()

def parse_asc_header(filepath: Path) -> dict:
    """Reads the 7-line header of a .ct (asc) file."""
    header = {}
    try:
        with open(filepath, 'r') as f:
            for i in range(config.ICE_ASC_HEADER_LINES):
                line = f.readline().strip().split()
                if len(line) == 2:
                    header[line[0].lower()] = float(line[1])
    except Exception as e:
        print(f"Error reading header from {filepath}: {e}")
        return {}
    return header

def load_ice_asc(date: datetime.date) -> xr.DataArray | None:
    """
    Loads a single 'ice asc' (.ct) file for a given date.
    
    *** FIX (v7) ***
    This version adds the coordinates and CRS back in.
    The training pipeline ('dataset.py') needs the CRS to exist
    so it can call reproject_to_master().
    
    The 'get_land_mask(v6)' function will still work because it
    only uses the .values from this function's output.
    """
    filename = f"g{date.strftime('%Y%m%d')}.ct"
    filepath = config.TRAIN_ICE_ASC_DIR / filename
    
    if not filepath.exists():
        # print(f"Warning: No ice asc file for {date}")
        return None
        
    header = parse_asc_header(filepath)
    if not header:
        print(f"Error: Could not parse header for {filepath}")
        return None
        
    data = np.loadtxt(filepath, skiprows=config.ICE_ASC_HEADER_LINES)
    
    # Handle potential size mismatch
    expected_size = header['ncols'] * header['nrows']
    if data.size != expected_size:
        print(f"Warning: Data size {data.size} does not match header {expected_size} in {filepath}. Reshaping...")
        try:
            data = data.flatten()[:int(expected_size)].reshape((int(header['nrows']), int(header['ncols'])))
        except Exception as e:
             print(f"Error reshaping data for {filepath}: {e}")
             return None

    # --- START REPLACEMENT ---
    # We are adding the coordinate and CRS logic back in.
    
    cellsize = header['cellsize']
    ncols = int(header['ncols'])
    nrows = int(header['nrows'])

    # 1. X Coordinates (Ascending, pixel-centered)
    x_start_center = header['xllcorner'] + (cellsize / 2)
    x_coords = np.arange(ncols) * cellsize + x_start_center

    # 2. Y Coordinates (Descending, pixel-centered)
    y_bottom_corner = header['yllcorner']
    y_top_corner = y_bottom_corner + (nrows * cellsize)
    y_start_center = y_top_corner - (cellsize / 2) # Center of top-most cell
    
    y_coords = np.arange(nrows) * -cellsize + y_start_center
    # --- END REPLACEMENT ---

    # Ensure data shape matches coordinate shape
    if data.shape != (len(y_coords), len(x_coords)):
         print(f"Error: Data shape {data.shape} mismatch with coords {(len(y_coords), len(x_coords))} in {filepath}")
         return None

    da = xr.DataArray(
        data,
        coords={'y': y_coords, 'x': x_coords},
        dims=["y", "x"],
        attrs={"crs": config.ICE_ASC_NATIVE_CRS} # EPSG:3175
    )
    
    # This is the line that fixes the error
    da = da.rio.set_spatial_dims(x_dim='x', y_dim='y')
    da.rio.write_crs(config.ICE_ASC_NATIVE_CRS, inplace=True)
    
    return da

def load_iceon_netcdf(date: datetime.date) -> xr.DataArray | None:
    """
    Loads a single 'ICECON' netcdf file for a given date.
    """
    search_path = str(config.TRAIN_ICECON_NC_DIR / f"*{date.strftime('%Y_%m_%d')}*.nc")
    files = glob.glob(search_path)
    
    if not files:
        # print(f"Warning: No ICECON file for {date}")
        return None
    
    try:
        with xr.open_dataset(files[0], engine="netcdf4") as ds:
            data_da = ds['ice_class'].load()
            data_da.name = 'ice_type'
            
            rename_dict = {}
            if 'longitude' in data_da.coords: rename_dict['longitude'] = 'x'
            if 'latitude' in data_da.coords: rename_dict['latitude'] = 'y'
            data_da = data_da.rename(rename_dict)
                
            data_da = data_da.rio.set_spatial_dims(x_dim='x', y_dim='y')
            data_da.rio.write_crs(config.MASTER_GRID_CRS, inplace=True)
            return data_da
            
    except Exception as e:
        print(f"Error loading ICECON file {files[0]}: {e}")
        return None

def load_glsea_data(date: datetime.date) -> xr.DataArray | None:
    """
    Loads GLSEA (water temp) data, selecting the slice for a given date.
    
    *** THIS IS THE FIX: This function no longer calls utils.get_land_mask() ***
    """
    try:
        with xr.open_dataset(config.TRAIN_GLSEA_NC_FILE, engine="netcdf4", decode_times=False) as ds:
            daily_data = ds['temp'].sel(time=date.strftime('%Y%m%d'), method='nearest').load()

            rename_dict = {}
            if 'longitude' in daily_data.coords: rename_dict['longitude'] = 'x'
            if 'latitude' in daily_data.coords: rename_dict['latitude'] = 'y'
            daily_data = daily_data.rename(rename_dict)
            
            daily_data = daily_data.rio.set_spatial_dims(x_dim='x', y_dim='y')
            daily_data.rio.write_crs(config.MASTER_GRID_CRS, inplace=True)
            return daily_data
    except Exception as e:
        print(f"Error loading GLSEA data for {date}: {e}")
        return None

def load_hrrr_data(date: datetime.date) -> xr.Dataset | None:
    """
    Loads HRRR weather data, selecting the slice for a given date.
    """
    try:
        with xr.open_dataset(config.TRAIN_HRRR_NC_FILE, engine="netcdf4") as ds:
            daily_data = ds.sel(time=date.strftime('%Y-%m-%d'), method='nearest').load()

            rename_dict = {}
            if 'nx' in daily_data.dims: rename_dict['nx'] = 'x'
            if 'ny' in daily_data.dims: rename_dict['ny'] = 'y'
            daily_data = daily_data.rename(rename_dict)

            if 'x' in daily_data.dims and 'y' in daily_data.dims:
                daily_data = daily_data.rio.set_spatial_dims(x_dim='x', y_dim='y')
            else:
                raise ValueError("Could not find standardized 'x' and 'y' dimensions in HRRR file.")

            if daily_data.rio.crs is None:
                print(f"Warning: HRRR file for {date} had no CRS. Writing default.")
                daily_data.rio.write_crs(config.ICE_ASC_NATIVE_CRS, inplace=True)
            
            return daily_data

    except Exception as e:
        print(f"Error loading HRRR data for {date}: {e}")
        return None

def load_icecon_class(date: datetime.date) -> xr.DataArray | None:
    """
    Loads the 'iceclass' variable from an ICECON .nc file for a given date.
    """
    date_str = date.strftime('%Y_%m_%d')
    search_pattern = str(config.TRAIN_ICECON_NC_DIR / f"*{date_str}*.nc")
    files = glob.glob(search_pattern)
    
    if not files:
        return None

    filepath = files[0]
    
    try:
        with xr.open_dataset(filepath, engine="netcdf4") as ds:
            ice_class_da = ds['iceclass'].load()

            rename_dict = {}
            if 'nx' in ice_class_da.dims:
                rename_dict['nx'] = 'x'
            if 'ny' in ice_class_da.dims:
                rename_dict['ny'] = 'y'
            ice_class_da = ice_class_da.rename(rename_dict)
            
            if 'x' in ice_class_da.dims and 'y' in ice_class_da.dims:
                ice_class_da = ice_class_da.rio.set_spatial_dims(x_dim='x', y_dim='y')
            else:
                raise ValueError("Could not find standardized 'x' and 'y' dimensions in ICECON file.")
            
            if ice_class_da.rio.crs is None:
                ice_class_da.rio.write_crs(config.MASTER_GRID_CRS, inplace=True)
            return ice_class_da
    except Exception as e:
        print(f"Error loading ICECON file {filepath}: {e}")
        return None

def load_glsea_water_temp(date: datetime.date) -> xr.DataArray | None:
    """
    Loads the GLSEA water temperature data for a given date.
    """
    try:
        with xr.open_dataset(config.TRAIN_GLSEA_NC_FILE, engine="netcdf4") as ds:
            daily_data = ds['temp'].sel(time=date.strftime('%Y-%m-%d'), method='nearest').load()

            rename_dict = {}
            if 'nx' in daily_data.dims:
                rename_dict['nx'] = 'x'
            if 'ny' in daily_data.dims:
                rename_dict['ny'] = 'y'
            daily_data = daily_data.rename(rename_dict)
            
            if 'x' in daily_data.dims and 'y' in daily_data.dims:
                daily_data = daily_data.rio.set_spatial_dims(x_dim='x', y_dim='y')
            else:
                raise ValueError("Could not find standardized 'x' and 'y' dimensions in GLSEA file.")
                    
            if daily_data.rio.crs is None:
                daily_data.rio.write_crs(config.MASTER_GRID_CRS, inplace=True)
            return daily_data
    except Exception as e:
        print(f"Error loading GLSEA data for {date}: {e}")
        return None