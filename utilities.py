import traceback
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
import numpy as np
import datetime
import warnings
import geopandas as gpd
from rasterio import features
from rasterio.transform import from_bounds
from pathlib import Path
import config

_master_grid_def = None
_land_mask = None
_shipping_route_mask = None

def get_master_grid() -> xr.DataArray:
    """
    Loads the master 2D grid definition by processing the raw GLSEA source files.
    This is slow, but necessary to work around a caching issue.
    """
    print("Generating master grid definition from raw GLSEA source files...")
    try:
        all_glsea_ice_das = []
        N_X = 1024
        N_Y = 1024
        lons = np.linspace(-92.5, -75.5, N_X)
        lats = np.linspace(41.0, 49.5, N_Y)

        for file_path in config.TRAIN_GLSEA_ICE_NC_FILES:
            if not file_path.exists():
                continue
            with xr.open_dataset(file_path) as ds:
                glsea_da = ds['temp']
                glsea_da = glsea_da.rename({'nx': 'y', 'ny': 'x'})
                glsea_da = glsea_da.assign_coords({'x': lons, 'y': lats})
                glsea_da = glsea_da.rio.set_spatial_dims(x_dim='x', y_dim='y')
                glsea_da.rio.write_crs(config.MASTER_GRID_CRS, inplace=True)
                all_glsea_ice_das.append(glsea_da)

        if not all_glsea_ice_das:
            raise FileNotFoundError("Could not find any GLSEA source files to generate master grid.")

        full_glsea_ice_dataset = xr.concat(all_glsea_ice_das, dim="time")
        master_da = full_glsea_ice_dataset.mean(dim='time', skipna=True).load().squeeze()
        
        print(f"Master grid generated. Shape: {master_da.shape}, CRS: {master_da.rio.crs}")
        return master_da

    except Exception as e:
        print(f"!!! FATAL Error in get_master_grid: {e}")
        traceback.print_exc()
        raise e

def get_land_mask(master_grid_2d: xr.DataArray) -> xr.DataArray:
    """
    Creates a land mask from the 2D GLSEA master grid.
    In GLSEA, land is NaN.
    """
    global _land_mask
    if _land_mask is not None:
        return _land_mask.copy()

    print("--- STARTING get_land_mask (from GLSEA grid) ---")

    # The data passed is already 2D
    glsea_data_2d = master_grid_2d

    # In GLSEA, land is NaN (isnull)
    land_mask_da = glsea_data_2d.isnull()

    # Convert boolean (True/False) to uint8 (1/0)
    land_mask_da = land_mask_da.astype(np.uint8)
    land_mask_da.name = "land_mask"

    _land_mask = land_mask_da.load()
    print("--- ENDING get_land_mask (from GLSEA grid - SUCCESS) ---")
    return _land_mask.copy()

def get_shipping_route_mask(master_grid: xr.DataArray) -> xr.DataArray:
    """
    Creates a binary mask of shipping routes.
    """
    global _shipping_route_mask
    if _shipping_route_mask is not None:
        return _shipping_route_mask.copy()

    print("--- STARTING get_shipping_route_mask ---")
    print(f"Loading shipping routes from: {config.SHIPPING_ROUTES_SHP}")
    # routes_gdf loads as EPSG:4326, which matches our master_grid.rio.crs
    routes_gdf = gpd.read_file(config.SHIPPING_ROUTES_SHP) 

    # Use the config CRS directly, as master_grid.rio.crs is unreliable
    if routes_gdf.crs != config.MASTER_GRID_CRS:
        print(f"Reprojecting routes from {routes_gdf.crs} to {config.MASTER_GRID_CRS}...")
        routes_gdf = routes_gdf.to_crs(config.MASTER_GRID_CRS)
    
    height, width = len(master_grid['y']), len(master_grid['x'])
    
    # We must re-project to a CRS that uses METERS for buffering.
    print(f"  > Projecting routes to {config.ICE_ASC_NATIVE_CRS} (meters) for buffering...")
    routes_gdf_proj = routes_gdf.to_crs(config.ICE_ASC_NATIVE_CRS)
    
    # Buffer by 2000 meters
    print("  > Buffering by 2000 meters...")
    shapes_proj = routes_gdf_proj.geometry.buffer(2000)
    
    # Project back to the master grid CRS (EPSG:4326)
    print(f"  > Projecting buffered routes back to {config.MASTER_GRID_CRS}...")
    shapes = shapes_proj.to_crs(config.MASTER_GRID_CRS)

    print("Rasterizing shipping routes to master grid...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning, lineno=387)
        route_mask_np = features.rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=master_grid.rio.transform(), # Use the grid's transform
            fill=0,
            default_value=1,
            dtype=np.uint8
        )

    route_mask_da = xr.DataArray(
        route_mask_np,
        coords={'y': master_grid['y'], 'x': master_grid['x']},
        dims=['y', 'x'],
        name="shipping_route_mask"
    )

    #route_mask_da.rio.write_crs(master_grid.rio.crs, inplace=True)
    route_mask_da.rio.write_crs(config.MASTER_GRID_CRS, inplace=True)
    _shipping_route_mask = route_mask_da
    print("--- ENDING get_shipping_route_mask (SUCCESS) ---")
    return _shipping_route_mask.copy()

def reproject_dataset(
    dataset: xr.Dataset,
    master_grid: xr.DataArray,
    resampling_method: Resampling = Resampling.bilinear
) -> xr.Dataset:
    """
    Reprojects an entire xarray.Dataset to a master grid.
    """
    reprojected_vars = []
    for var_name in dataset.data_vars:
        data_arr = dataset[var_name]
        
        # Ensure the DataArray has spatial attributes for reprojection
        if data_arr.rio.crs is None:
            # Attempt to set a CRS if it's missing, assuming WGS84
            data_arr.rio.write_crs("EPSG:4326", inplace=True)

        var_reprojected = data_arr.rio.reproject_match(
            master_grid,
            resampling=resampling_method
        )
        reprojected_vars.append(var_reprojected)
        
    return xr.merge(reprojected_vars)

def print_da_stats(data: np.ndarray, name: str):
    """Prints a quick summary of a numpy array."""
    print(f"  > Stats for '{name}':")
    if data.size == 0:
        print("    Array is empty!")
        return
    print(f"    Shape: {data.shape}")
    print(f"    Min:   {np.nanmin(data):.4f}")
    print(f"    Max:   {np.nanmax(data):.4f}")
    print(f"    Mean:  {np.nanmean(data):.4f}")
    print(f"    NaNs:  {np.isnan(data).sum()}")

def get_land_mask_from_test_ice() -> xr.DataArray:
    """
    Creates a land mask from the test ice file.
    This is to get a correct land mask.
    """
    print("--- STARTING get_land_mask_from_test_ice ---")
    try:
        with xr.open_dataset(config.TEST_ICE_NC, decode_times=False) as ds:
            ice_da = ds['ice_cover'].load().squeeze()
            
            # The land is where the data is NaN
            land_mask_da = ice_da.isnull()
            
            # Convert boolean (True/False) to uint8 (1/0)
            land_mask_da = land_mask_da.astype(np.uint8)
            land_mask_da.name = "land_mask"
            
            # Rename dims to 'x' and 'y'
            rename_dict = {}
            if 'lon' in land_mask_da.coords: rename_dict['lon'] = 'x'
            if 'lat' in land_mask_da.coords: rename_dict['lat'] = 'y'
            land_mask_da = land_mask_da.rename(rename_dict)

            # We need to assign the master grid coordinates to this mask
            master_grid = get_master_grid()
            land_mask_da.values = np.flipud(land_mask_da.values).copy()
            land_mask_da = land_mask_da.assign_coords({
                'x': master_grid.coords['x'],
                'y': master_grid.coords['y']
            })
            land_mask_da = land_mask_da.rio.set_spatial_dims(x_dim='x', y_dim='y')
            land_mask_da.rio.write_crs(config.MASTER_GRID_CRS, inplace=True)

            print("--- ENDING get_land_mask_from_test_ice (SUCCESS) ---")
            return land_mask_da

    except Exception as e:
        print(f"!!! FATAL Error in get_land_mask_from_test_ice: {e}")
        traceback.print_exc()
        raise e