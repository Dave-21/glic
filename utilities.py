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
import data_loaders
from pathlib import Path

import config

_master_grid_def = None
_land_mask = None
_shipping_route_mask = None

def get_master_grid_definition() -> xr.DataArray:
    """
    Loads the master 2D grid by getting the full GLSEA data
    (which is now georeferenced) and taking the first time slice.
    """
    global _master_grid_def
    if _master_grid_def is not None:
        return _master_grid_def.copy()

    print("Loading master grid definition (from GLSEA cache)...")
    try:
        # Get the full, georeferenced data
        glsea_data = data_loaders.get_glsea_ice_data()

        # Use the first time slice as our master 2D grid
        master_da = glsea_data.isel(time=0).load().squeeze()

        _master_grid_def = master_da.copy()

        print(f"Master grid loaded. Shape: {master_da.shape}, CRS: {master_da.rio.crs}")
        # These should now be Lat/Lon values
        print(f"Grid X (Lon) (min/max): {master_da['x'].min().item()} / {master_da['x'].max().item()}")
        print(f"Grid Y (Lat) (min/max): {master_da['y'].min().item()} / {master_da['y'].max().item()}")
        return _master_grid_def

    except Exception as e:
        print(f"!!! FATAL Error in get_master_grid_definition: {e}")
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

    if routes_gdf.crs != master_grid.rio.crs:
        print(f"Reprojecting routes from {routes_gdf.crs} to {master_grid.rio.crs}...")
        routes_gdf = routes_gdf.to_crs(master_grid.rio.crs)
    
    height, width = len(master_grid['y']), len(master_grid['x'])
    
    # We must re-project to a CRS that uses METERS for buffering.
    print(f"  > Projecting routes to {config.ICE_ASC_NATIVE_CRS} (meters) for buffering...")
    routes_gdf_proj = routes_gdf.to_crs(config.ICE_ASC_NATIVE_CRS)
    
    # Buffer by 2000 meters
    print("  > Buffering by 2000 meters...")
    shapes_proj = routes_gdf_proj.geometry.buffer(2000)
    
    # Project back to the master grid CRS (EPSG:4326)
    print(f"  > Projecting buffered routes back to {master_grid.rio.crs}...")
    shapes = shapes_proj.to_crs(master_grid.rio.crs)

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

    route_mask_da.rio.write_crs(master_grid.rio.crs, inplace=True)
    _shipping_route_mask = route_mask_da
    print("--- ENDING get_shipping_route_mask (SUCCESS) ---")
    return _shipping_route_mask.copy()
