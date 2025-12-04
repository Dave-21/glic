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
import pandas as pd
import config
import matplotlib.pyplot as plt

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
                glsea_da = ds["temp"]
                glsea_da = glsea_da.rename({"nx": "y", "ny": "x"})
                glsea_da = glsea_da.assign_coords({"x": lons, "y": lats})
                glsea_da = glsea_da.rio.set_spatial_dims(x_dim="x", y_dim="y")
                glsea_da.rio.write_crs(config.MASTER_GRID_CRS, inplace=True)
                all_glsea_ice_das.append(glsea_da)

        if not all_glsea_ice_das:
            raise FileNotFoundError(
                "Could not find any GLSEA source files to generate master grid."
            )

        full_glsea_ice_dataset = xr.concat(all_glsea_ice_das, dim="time")
        master_da = (
            full_glsea_ice_dataset.mean(dim="time", skipna=True).load().squeeze()
        )

        print(
            f"Master grid generated. Shape: {master_da.shape}, CRS: {master_da.rio.crs}"
        )
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
    
    if config.DEBUG_MODE:
        config.DEBUG_DIR.mkdir(exist_ok=True)
        plt.figure(figsize=(10, 10))
        _land_mask.plot()
        plt.title("Land Mask")
        plt.savefig(config.DEBUG_DIR / "land_mask.png")
        plt.close()
        print(f"DEBUG: Saved land mask visualization to {config.DEBUG_DIR / 'land_mask.png'}")

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
        print(
            f"Reprojecting routes from {routes_gdf.crs} to {config.MASTER_GRID_CRS}..."
        )
        routes_gdf = routes_gdf.to_crs(config.MASTER_GRID_CRS)

    height, width = len(master_grid["y"]), len(master_grid["x"])

    # We must re-project to a CRS that uses METERS for buffering.
    print(
        f"  > Projecting routes to {config.ICE_ASC_NATIVE_CRS} (meters) for buffering..."
    )
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
            transform=master_grid.rio.transform(),  # Use the grid's transform
            fill=0,
            default_value=1,
            dtype=np.uint8,
        )

    route_mask_da = xr.DataArray(
        route_mask_np,
        coords={"y": master_grid["y"], "x": master_grid["x"]},
        dims=["y", "x"],
        name="shipping_route_mask",
    )

    # route_mask_da.rio.write_crs(master_grid.rio.crs, inplace=True)
    route_mask_da.rio.write_crs(config.MASTER_GRID_CRS, inplace=True)
    _shipping_route_mask = route_mask_da
    
    if config.DEBUG_MODE:
        config.DEBUG_DIR.mkdir(exist_ok=True)
        plt.figure(figsize=(10, 10))
        _shipping_route_mask.plot()
        plt.title("Shipping Route Mask")
        plt.savefig(config.DEBUG_DIR / "shipping_route_mask.png")
        plt.close()
        print(f"DEBUG: Saved shipping route visualization to {config.DEBUG_DIR / 'shipping_route_mask.png'}")

    print("--- ENDING get_shipping_route_mask (SUCCESS) ---")
    return _shipping_route_mask.copy()


def reproject_dataset(
    dataset: xr.Dataset,
    master_grid: xr.DataArray,
    resampling_method: Resampling = Resampling.bilinear,
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
            master_grid, resampling=resampling_method
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
            ice_da = ds["ice_cover"].load().squeeze()

            # The land is where the data is NaN
            land_mask_da = ice_da.isnull()

            # Convert boolean (True/False) to uint8 (1/0)
            land_mask_da = land_mask_da.astype(np.uint8)
            land_mask_da.name = "land_mask"

            # Rename dims to 'x' and 'y'
            rename_dict = {}
            if "lon" in land_mask_da.coords:
                rename_dict["lon"] = "x"
            if "lat" in land_mask_da.coords:
                rename_dict["lat"] = "y"
            land_mask_da = land_mask_da.rename(rename_dict)

            # We need to assign the master grid coordinates to this mask
            master_grid = get_master_grid()
            land_mask_da.values = np.flipud(land_mask_da.values).copy()
            land_mask_da = land_mask_da.assign_coords(
                {"x": master_grid.coords["x"], "y": master_grid.coords["y"]}
            )
            land_mask_da = land_mask_da.rio.set_spatial_dims(x_dim="x", y_dim="y")
            land_mask_da.rio.write_crs(config.MASTER_GRID_CRS, inplace=True)

            print("--- ENDING get_land_mask_from_test_ice (SUCCESS) ---")
            return land_mask_da

    except Exception as e:
        print(f"!!! FATAL Error in get_land_mask_from_test_ice: {e}")
        traceback.print_exc()
        raise e

# --- SIGRID-3 Code Mappings and Functions ---

# Represents the mean thickness in METERS for a given Stage of Development code.
SIGRID3_THICKNESS_MAPPING_METERS = {
    "00": 0.0,
    "01": 0.0,
    "80": 0.0,
    "81": 0.05,
    "82": 0.125,
    "83": 0.225,
    "84": 0.125,
    "85": 0.225,
    "86": 0.75,
    "87": 0.50,
    "88": 0.40,
    "89": 0.60,
    "91": 0.95,
    "93": 1.60,
    "95": 2.0,
    "96": 2.0,
    "97": 2.5,
    "98": 0.0,
    "99": 0.0,
    "-9": 0.0,
}

# Placeholder for SIGRID3_FLOE_SIZE_MAPPING as it was not provided in the original context
# but is referenced by get_floe_size_val.
SIGRID3_FLOE_SIZE_MAPPING = {
    # Example values, replace with actual SIGRID-3 floe size mappings
    "1": 0.1, # Pancake ice, shuga
    "2": 0.2, # Small floe
    "3": 0.3, # Medium floe
    "4": 0.4, # Big floe
    "5": 0.5, # Vast floe
    "6": 0.6, # Giant floe
    "7": 0.7, # Fast ice
    "8": 0.8, # Iceberg
    "9": 0.9, # Bergy bit
    "0": 0.0, # No ice
    "-9": 0.0, # Missing
}


def get_floe_size_val(code):
    """Parses a SIGRID-3 floe size code to a normalized 0-1 value."""
    if pd.isna(code):
        return 0.0
    s_code = str(code).split(".")[0].strip()
    return SIGRID3_FLOE_SIZE_MAPPING.get(s_code, 0.0)

def get_thickness_class(thickness_m: float) -> int:
    """
    Bins continuous thickness (meters) into SIGRID-3 classes.
    0: Water (0)
    1: New Ice (<0.10)
    2: Young Ice (0.10 - 0.30)
    3: First Year Thin (0.30 - 0.70)
    4: First Year Medium (0.70 - 1.20)
    5: First Year Thick (>1.20)
    """
    if thickness_m <= 0.001: return 0
    if thickness_m < 0.10: return 1
    if thickness_m < 0.30: return 2
    if thickness_m < 0.70: return 3
    if thickness_m < 1.20: return 4
    return 5

def get_thickness_val(code):
    """Parses a SIGRID-3 thickness code to a numeric value in meters."""
    if pd.isna(code):
        return 0.0
    s_code = str(code).split(".")[0].strip()
    return SIGRID3_THICKNESS_MAPPING_METERS.get(s_code, 0.0)

def get_conc_fraction(code):
    """
    Parses a SIGRID-3 concentration code to a fractional value (0.0 to 1.0).
    """
    if pd.isna(code):
        return 0.0
    s_code = str(code).split(".")[0].strip()

    try:
        val = int(s_code)
        if 0 <= val <= 10:
            return val / 10.0
        if val == 91:
            return 0.95
        if val == 92:
            return 1.0
        if 10 < val < 90:
            low = val // 10
            high = val % 10
            return ((low + high) / 2.0) / 10.0
        return 0.0
    except (ValueError, TypeError):
        return 0.0

def calculate_weighted_thickness(row: pd.Series) -> float:
    """
    Calculates a weighted average thickness for a polygon.
    """
    th_a = get_thickness_val(row.get("SA"))
    cn_a = get_conc_fraction(row.get("CA"))
    th_b = get_thickness_val(row.get("SB"))
    cn_b = get_conc_fraction(row.get("CB"))
    th_c = get_thickness_val(row.get("SC"))
    cn_c = get_conc_fraction(row.get("CC"))

    total_conc = cn_a + cn_b + cn_c
    if total_conc == 0:
        return 0.0

    weighted_total = (th_a * cn_a) + (th_b * cn_b) + (th_c * cn_c)
    avg_thickness = weighted_total / total_conc
    final_thickness = avg_thickness * get_conc_fraction(row.get("CT"))
    return min(final_thickness, 5.0)