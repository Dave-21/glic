import xarray as xr
import rioxarray
from rasterio.enums import Resampling
import numpy as np
import datetime
from typing import Tuple
import pandas as pd
import geopandas as gpd
from rasterio import features
import warnings
import cartopy.crs as ccrs
import bathyreq
import traceback
import config
import utilities
import s3fs
import zarr


# --- Cache Variables ---
_hrrr_day_cache = {}
# _nic_ice_data_cache = None # This is not used
_hrrr_grid_cache = None
# _glsea_ice_data_cache = None # This will be removed
_gebco_data_cache = None
_nic_shp_cache = {}
# _hrrr_train_dataset_cache = None # This will be removed

# --- HRRR Projection (from HRRR Guide) ---
HRRR_PROJECTION = ccrs.LambertConformal(
    central_longitude=262.5,
    central_latitude=38.5,
    standard_parallels=(38.5, 38.5),
    globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229),
)


def download_and_reproject_hrrr_for_day(
    target_date: datetime.date, master_grid: xr.DataArray
) -> xr.Dataset | None:
    """
    Downloads HRRR data for a single day from the Zarr store, selects variables,
    and reprojects it to the master grid.
    """
    try:
        s3 = s3fs.S3FileSystem(anon=True)
        date_str = target_date.strftime("%Y%m%d")
        zarr_path = (
            f"{config.TRAIN_HRRR_ZARR_ROOT}/sfc/{date_str}/{date_str}_00z_anl.zarr"
        )
        s3_map = s3fs.S3Map(zarr_path, s3=s3)
        ds = xr.open_zarr(s3_map)

        # Select and rename variables
        vars_to_load = {}
        for key, path in config.HRRR_VAR_MAP.items():
            # The path is 'VAR_NAME/level' but the zarr has 'level/VAR_NAME'
            level, var_name = path.split("/")
            vars_to_load[key] = ds[level][var_name]

        hrrr_ds = xr.Dataset(vars_to_load)
        hrrr_ds = hrrr_ds.rio.write_crs(HRRR_PROJECTION)

        # Reproject to master grid
        reprojected_ds = utilities.reproject_dataset(hrrr_ds, master_grid)
        return reprojected_ds

    except Exception as e:
        print(f"!!! ERROR processing HRRR for {target_date}: {e}")
        return None


def load_and_rasterize_nic_shp(
    target_date: datetime.date, master_grid: xr.DataArray
) -> Tuple[xr.DataArray, xr.DataArray] | None:
    """
    Loads a NIC shapefile for a given date, reprojects it, and rasterizes
    the ice concentration (CT) and thickness (SA) onto the master grid.
    """
    if config.DEBUG_MODE:
        print(f"DEBUG: Loading NIC shapefile for date: {target_date}")
        
    # 1. Check cache first
    if target_date in _nic_shp_cache:
        return _nic_shp_cache[target_date]

    # 2. Find the correct shapefile
    date_str_yyyymmdd = target_date.strftime("%Y%m%d")
    date_str_yymmdd = target_date.strftime("%y%m%d")

    # Search for files like 'GLyymmdd.shp' or 'glYYYYMMDD_lam.shp' etc.
    # The glob is case-insensitive.
    search_pattern = f"*{date_str_yymmdd}*.shp"

    found_files = list(config.TRAIN_NIC_SHP_DIR.rglob(search_pattern))

    if not found_files:
        # Fallback search for different naming conventions if needed
        search_pattern_alt = f"*{date_str_yyyymmdd}*.shp"
        found_files = list(config.TRAIN_NIC_SHP_DIR.rglob(search_pattern_alt))

    if not found_files:
        print(f"!!! WARNING: No NIC shapefile found for {target_date}. Returning None.")
        return None

    shp_path = found_files[0]

    try:
        # 3. Read and Reproject
        gdf = gpd.read_file(shp_path)
        gdf_reprojected = gdf.to_crs(master_grid.rio.crs)

        # 4. Prepare for Rasterization
        transform = master_grid.rio.transform()
        out_shape = master_grid.shape

        # --- Create geometry-value pairs for concentration and thickness ---

        # Use CT for Total Concentration if available, otherwise sum of partials
        # Default to 'CT' if it exists, otherwise use 'A_ICE' or another fallback
        conc_col = (
            "CT" if "CT" in gdf_reprojected.columns else "A_ICE"
        )  # Common alternative name
        if conc_col not in gdf_reprojected.columns:
            # if neither is present, we cannot proceed
            raise ValueError(
                f"Shapefile {shp_path.name} missing concentration column ('CT' or 'A_ICE')."
            )

        shapes_conc = []
        for _, row in gdf_reprojected.iterrows():
            conc_val = utilities.get_conc_fraction(row.get(conc_col))
            shapes_conc.append((row["geometry"], conc_val))

        # Use SA for Stage of Development (Thickness proxy)
        thick_col = (
            "SA" if "SA" in gdf_reprojected.columns else "S_ICE"
        )  # Common alternative
        if thick_col not in gdf_reprojected.columns:
            raise ValueError(
                f"Shapefile {shp_path.name} missing thickness column ('SA' or 'S_ICE')."
            )

        shapes_thick = []
        for _, row in gdf_reprojected.iterrows():
            thick_val_m = utilities.get_thickness_val(row.get(thick_col))
            shapes_thick.append((row["geometry"], thick_val_m))

        # 5. Rasterize
        conc_raster = features.rasterize(
            shapes=shapes_conc,
            out_shape=out_shape,
            transform=transform,
            fill=0.0,
            dtype="float32",
        )
        thick_raster = features.rasterize(
            shapes=shapes_thick,
            out_shape=out_shape,
            transform=transform,
            fill=0.0,
            dtype="float32",
        )

        # 6. Convert to DataArray
        da_conc = xr.DataArray(
            conc_raster,
            coords=master_grid.coords,
            dims=master_grid.dims,
            name="ice_concentration_nic",
        )
        da_thick = xr.DataArray(
            thick_raster,
            coords=master_grid.coords,
            dims=master_grid.dims,
            name="ice_thickness_nic",
        )

        # Set CRS for the new DataArrays
        da_conc.rio.write_crs(master_grid.rio.crs, inplace=True)
        da_thick.rio.write_crs(master_grid.rio.crs, inplace=True)

        # 7. Cache and return
        result = (da_conc, da_thick)
        _nic_shp_cache[target_date] = result

        return result

    except Exception as e:
        print(f"!!! ERROR processing shapefile {shp_path.name}: {e}")
        traceback.print_exc()
        return None


def get_consolidated_hrrr_dataset() -> xr.Dataset | None:
    """
    Opens the pre-generated consolidated HRRR training dataset from disk
    without loading it into memory.
    """
    if not config.TRAIN_HRRR_NC_FILE.exists():
        print("!!! ERROR: Consolidated HRRR training file not found!")
        print(f"    Expected at: {config.TRAIN_HRRR_NC_FILE}")
        print("    Please run 'python setup.py' to generate it.")
        return None

    print(
        f"Opening consolidated HRRR training data from: {config.TRAIN_HRRR_NC_FILE}..."
    )
    try:
        # Open the dataset without loading it into memory.
        # Chunking can be useful here if the underlying file is not chunked appropriately.
        ds = xr.open_dataset(
            config.TRAIN_HRRR_NC_FILE, engine="netcdf4", chunks={"time": 1}
        )
        print("Consolidated HRRR data opened for lazy loading.")
        return ds
    except Exception as e:
        print(f"!!! ERROR: Failed to open consolidated HRRR file: {e}")
        return None


def load_hrrr_data_for_day(
    target_date: datetime.date, master_grid: xr.DataArray
) -> xr.Dataset | None:
    """
    Loads HRRR data for a single day from the pre-consolidated NetCDF file.
    """
    if config.DEBUG_MODE:
        print(f"DEBUG: Loading HRRR data for day: {target_date}")
        
    if target_date in _hrrr_day_cache:
        return _hrrr_day_cache[target_date].copy()

    consolidated_ds = get_consolidated_hrrr_dataset()
    if consolidated_ds is None:
        return None

    try:
        # Using .sel() with a date string and method='nearest' is robust
        daily_ds = consolidated_ds.sel(
            time=str(target_date), method="nearest", tolerance=pd.Timedelta(hours=12)
        )

        # Cache and return
        daily_ds.load()  # Ensure this slice is in memory
        _hrrr_day_cache[target_date] = daily_ds
        print(f"--- Successfully loaded HRRR for {target_date} from local file ---")
        return daily_ds.copy()
    except KeyError:
        print(
            f"!!! WARNING: Could not find HRRR data for {target_date} in the consolidated file."
        )
        return None
    except Exception as e:
        print(
            f"!!! ERROR: An unexpected error occurred while loading HRRR for {target_date}: {e}"
        )
        return None


def load_glsea_ice_data() -> xr.DataArray:
    """
    Loads and processes GLSEA ice data from NetCDF files.
    Combines 2018 and 2019 data, applies the rotation fix, and
    *manually assigns real-world Lat/Lon coordinates* to georeference it.
    """
    # 1. Disk Cache Check
    if config.TRAIN_GLSEA_ICE_CACHE_FILE.exists():
        print(
            f"Opening GLSEA ice data from disk cache: {config.TRAIN_GLSEA_ICE_CACHE_FILE}"
        )
        try:
            # Open the dataset without loading it into memory.
            ds = xr.open_dataset(
                config.TRAIN_GLSEA_ICE_CACHE_FILE, engine="netcdf4", chunks={"time": 10}
            )
            print("GLSEA ice data opened for lazy loading.")
            return ds["glsea_ice_temp"]
        except Exception as e:
            print(
                f"Error opening GLSEA ice cache file ({e}), will attempt to regenerate."
            )

    print("Loading and processing GLSEA ice data (this is a slow, one-time process)...")

    all_glsea_ice_das = []

    # We are defining the grid's extent in Lat/Lon (EPSG:4326)
    # Based on the rotation, 'ny' (1024) is 'x' (Longitude)
    # and 'nx' (1024) is 'y' (Latitude)
    N_X = 1024  # ny
    N_Y = 1024  # nx

    # Longitude (x-axis)
    lons = np.linspace(-92.5, -75.5, N_X)  # West to East
    # Latitude (y-axis)
    # lats = np.linspace(49.5, 41.0, N_Y)  # North to South (descending)
    lats = np.linspace(41.0, 49.5, N_Y)  # South to North (ascending)

    for file_path in config.TRAIN_GLSEA_ICE_NC_FILES:
        if not file_path.exists():
            print(f"!!! WARNING: GLSEA ice file not found: {file_path}, skipping.")
            continue

        try:
            # with xr.open_dataset(str(file_path)) as ds:
            with xr.open_dataset(str(file_path), chunks={"time": 1}) as ds:
                glsea_da = ds["temp"]  # shape (time, nx, ny)

                # 1. Rename dims: 'nx' -> 'y', 'ny' -> 'x'
                #    This makes the dims ('time', 'y', 'x')
                glsea_da = glsea_da.rename({"nx": "y", "ny": "x"})

                # 2. Assign the Lat/Lon coordinates we defined
                glsea_da = glsea_da.assign_coords({"x": lons, "y": lats})

                # --- SET SPATIAL DIMS & CRS ---
                # 3. Tell rioxarray what the dims and CRS are
                glsea_da = glsea_da.rio.set_spatial_dims(x_dim="x", y_dim="y")
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
    # full_glsea_ice_dataset.load()

    print(f"Caching GLSEA ice data to: {config.TRAIN_GLSEA_ICE_CACHE_FILE}")
    config.TRAIN_GLSEA_ICE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    ds_to_save = full_glsea_ice_dataset.to_dataset(name="glsea_ice_temp")
    # ds_to_save.to_netcdf(config.TRAIN_GLSEA_ICE_CACHE_FILE, engine="h5netcdf")
    ds_to_save.to_netcdf(
        config.TRAIN_GLSEA_ICE_CACHE_FILE, engine="h5netcdf", compute=True
    )

    if config.DEBUG_MODE:
        print("--- DEBUG: Master GLSEA Ice Data Stats ---")
        print(f"  Shape: {full_glsea_ice_dataset.shape}")
        print(f"  Min:   {full_glsea_ice_dataset.min().compute().item():.4f}")
        print(f"  Max:   {full_glsea_ice_dataset.max().compute().item():.4f}")
        print(f"  Mean:  {full_glsea_ice_dataset.mean().compute().item():.4f}")
        # For NaNs, we need to compute the values first or use a dask-friendly method
        # np.isnan(dask_array) returns a dask array. .sum() returns a dask array.
        nan_count = np.isnan(full_glsea_ice_dataset).sum().compute().item()
        print(f"  NaNs:  {nan_count} (Land pixels)")
        print("-----------------------------------------------------")

    print("Re-opening from the new cache file for lazy loading...")
    ds = xr.open_dataset(
        config.TRAIN_GLSEA_ICE_CACHE_FILE, engine="netcdf4", chunks={"time": 10}
    )

    return ds["glsea_ice_temp"]


def load_gebco_data(master_grid: xr.DataArray) -> xr.DataArray:
    """
    Loads and processes GEBCO bathymetry data using bathyreq.
    """
    if config.DEBUG_MODE:
        print(f"DEBUG: Loading GEBCO data")
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
            print(
                f"Error loading GEBCO cache file ({e}), re-running expensive computation."
            )

    print("Loading and processing GEBCO data (this is a slow, one-time process)...")

    try:
        # Download the data using bathyreq
        req = bathyreq.BathyRequest()
        data, lon, lat = req.get_area(
            longitude=[-93.1, -74.9], latitude=[40.9, 49.6], resolution="20m"
        )

        # Create an xarray DataArray from the downloaded data
        da = xr.DataArray(
            data,
            coords={"y": lat, "x": lon},
            dims=["y", "x"],
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
    if config.DEBUG_MODE:
        print(f"DEBUG: GEBCO data shape: {_gebco_data_cache.shape}")
    return _gebco_data_cache.copy()


"""
Test Data Loaders
"""


def load_test_ice_raw() -> xr.DataArray | None:
    """
    Loads the test set initial condition for ice concentration (RAW).
    Converts 0-100 to 0-1 for consistency, but keeps land as NaN for now.
    """
    if config.DEBUG_MODE:
        print(f"DEBUG: Loading raw test ice data from: {config.TEST_ICE_NC}")
        
    try:
        with xr.open_dataset(
            config.TEST_ICE_NC, engine="netcdf4", decode_times=False
        ) as ds:
            data_da = ds["ice_cover"].load().squeeze()  # Values 0-100, NaN for land
            
            if config.DEBUG_MODE:
                print("--- DEBUG: Raw Test Ice Data Stats ---")
                print(f"  Shape: {data_da.shape}")
                print(f"  Min:   {np.nanmin(data_da.values):.4f}")
                print(f"  Max:   {np.nanmax(data_da.values):.4f}")
                print(f"  Mean:  {np.nanmean(data_da.values):.4f}")
                print(f"  NaNs:  {np.isnan(data_da.values).sum()} (Land pixels)")
                print("-----------------------------------------")

            # --- 1. Rename dims to 'x' and 'y' ---
            rename_dict = {}
            if "lon" in data_da.coords:
                rename_dict["lon"] = "x"
            if "lat" in data_da.coords:
                rename_dict["lat"] = "y"
            data_da = data_da.rename(rename_dict)

            # --- 2. Steal coordinates from the master grid ---
            print("...Assigning master grid coordinates to test ice data...")
            master_grid = utilities.get_master_grid()

            # Flip data to match ascending 'y' coords
            data_da.values = np.flipud(data_da.values)

            data_da = data_da.assign_coords(
                {"x": master_grid.coords["x"], "y": master_grid.coords["y"]}
            )

            # --- 3. Set spatial dims & CRS ---
            data_da = data_da.rio.set_spatial_dims(x_dim="x", y_dim="y")
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
    if config.DEBUG_MODE:
        print(f"DEBUG: Loading raw test water temp data from: {config.TEST_GLSEA_NC}")
    try:
        with xr.open_dataset(
            config.TEST_GLSEA_NC, engine="netcdf4", decode_times=False
        ) as ds:
            data_da = ds["sst"].load().squeeze()  # Values in C, NaN for land/ice

            # --- 1. Rename dims to 'x' and 'y' ---
            rename_dict = {}
            if "lon" in data_da.coords:
                rename_dict["lon"] = "x"
            if "lat" in data_da.coords:
                rename_dict["lat"] = "y"
            data_da = data_da.rename(rename_dict)

            # --- 2. Steal coordinates from the master grid ---
            print("...Assigning master grid coordinates to test water temp data...")
            master_grid = utilities.get_master_grid()

            # Flip data to match ascending 'y' coords
            data_da.values = np.flipud(data_da.values)

            data_da = data_da.assign_coords(
                {"x": master_grid.coords["x"], "y": master_grid.coords["y"]}
            )

            # --- 3. Set spatial dims & CRS ---
            data_da = data_da.rio.set_spatial_dims(x_dim="x", y_dim="y")
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
    if config.DEBUG_MODE:
        print(f"DEBUG: Loading raw HRRR forecast data from: {config.TEST_HRRR_NC}")
    try:
        with xr.open_dataset(
            config.TEST_HRRR_NC, engine="netcdf4", decode_times=False
        ) as ds:
            ds_load = ds.load()

            rename_dict = {}
            if "lon" in ds_load.coords:
                rename_dict["lon"] = "x"
            if "lat" in ds_load.coords:
                rename_dict["lat"] = "y"
            ds_load = ds_load.rename(rename_dict)

            # --- 0-360 longitude ---
            if "x" in ds_load.coords and ds_load["x"].max() > 180:
                print("...Converting HRRR longitude from 0-360 to -180-180...")
                ds_load.coords["x"] = ((ds_load.coords["x"] + 180) % 360) - 180
                ds_load = ds_load.sortby("x")

            ds_load = ds_load.rio.set_spatial_dims(x_dim="x", y_dim="y")

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
    if config.DEBUG_MODE:
        print(f"DEBUG: Loading ground truth data for dates: {dates}")
    import data_loaders
    import pandas as pd

    master_ice_and_temp_data = data_loaders.load_glsea_ice_data()
    ground_truth_data = {}

    for date in dates:
        date_pd = pd.to_datetime(date)
        da = master_ice_and_temp_data.sel(time=date_pd, method="nearest")
        time_diff = abs((da.time.values - date_pd.to_numpy()) / np.timedelta64(1, "h"))

        if time_diff <= 12:
            da_squeezed = da.squeeze()
            ice_conc = xr.where(da_squeezed < 0, -da_squeezed, 0.0)
            ground_truth_data[date] = ice_conc
        else:
            ground_truth_data[date] = None

    return ground_truth_data