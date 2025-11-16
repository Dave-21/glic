import xarray as xr
import rioxarray
import numpy as np
import config
import utilities

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
            master_grid = utilities.get_master_grid_definition()
            
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
            master_grid = utilities.get_master_grid_definition()
            
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
            
            # --- Fix for 0-360 longitude ---
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