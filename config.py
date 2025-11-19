import os
from pathlib import Path
import datetime

# --- Project Root ---
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = PROJECT_ROOT / "datasets"

START_DATE = datetime.date(2018, 12, 1)
END_DATE = datetime.date(2019, 3, 1)

# This maps our standard names to the *full Zarr paths*
# within the 2018-2019 files.
HRRR_VAR_MAP = {
    'air_temp_2m':    '2m_above_ground/TMP',
    'u_wind_10m':     '10m_above_ground/UGRD',
    'v_wind_10m':     '10m_above_ground/VGRD',
    'precip_surface': 'surface/PRATE',
}
# This list (e.g., ['air_temp_2m', ...]) is used by the rest of the scripts
HRRR_VARS = list(HRRR_VAR_MAP.keys())

TRAIN_NIC_SHP_DIR = DATA_ROOT / "Ice Data" / "nic_shapefiles_unzipped"
# Zarr URL for HRRR data
# This is the root bucket, as shown in the guide
TRAIN_HRRR_ZARR_ROOT = 's3://hrrrzarr' 
# This is the grid file, from "Option 2" of the guide

# Path to the unzipped shipping routes shapefile
SHIPPING_ROUTES_SHP = DATA_ROOT / "Shipping_Routes" / "shippinglanes.shp"
# Land mask file
TRAIN_ICE_ASC_DIR = DATA_ROOT / "Ice Data" / "ice asc" # Used for land mask
LAND_MASK_FILE = TRAIN_ICE_ASC_DIR / "g20190111.ct" # Or any single file for the grid
TRAIN_NIC_CACHE_FILE = DATA_ROOT / "Ice Data" / "nic_ice_data_cache.nc"

TRAIN_GLSEA_ICE_NC_FILES = [
    DATA_ROOT / "Water Surface Temperature Data" / "GLSEA_ICE" / "2018_glsea_ice.nc",
    DATA_ROOT / "Water Surface Temperature Data" / "GLSEA_ICE" / "2019_glsea_ice.nc",
]
TRAIN_GLSEA_ICE_CACHE_FILE = DATA_ROOT / "Water Surface Temperature Data" / "glsea_ice_data_cache.nc"
GEBCO_FILE = DATA_ROOT / "Water Surface Temperature Data" / "gebco_great_lakes.tif"

# --- Master Grid Definition (from training data) ---
ORIGINAL_GLSEA_FILE = DATA_ROOT / "Water Surface Temperature Data" / "netcdf" / "glsea_20190111-20190131.nc"
MASTER_GRID_TEMPLATE_FILE = ORIGINAL_GLSEA_FILE
MASTER_GRID_CRS = "EPSG:4326"  # Our target grid (standard Lat/Lon)
ICE_ASC_NATIVE_CRS = "EPSG:3175" # NAD83 / Great Lakes Albers (projected, in meters)
GLSEA_VARIABLE_NAME = "temp"

# --- Training Constants ---
ICE_ASC_HEADER_LINES = 7
ICE_ASC_NODATA_VAL = -1

TEST_DATA_DIR = DATA_ROOT / "Test Data"
TEST_IC_DIR = TEST_DATA_DIR / "Ice & Water Surface Temperature Initial Conditions"
TEST_WEATHER_DIR = TEST_DATA_DIR / "Weather Data"

TEST_ICE_NC = TEST_IC_DIR / "ice_test_initial_condition.nc"
TEST_GLSEA_NC = TEST_IC_DIR / "glsea_ice_test_initial_condition.nc"
TEST_HRRR_NC = TEST_WEATHER_DIR / "hrrr_weather_test_period.nc"
TEST_GFS_NC = TEST_WEATHER_DIR / "gfs_weather_test_period.nc"

# --- Output Paths ---
MODEL_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pth"
OUTPUT_DIR = PROJECT_ROOT / "forecasts"
FORECAST_FILE = OUTPUT_DIR / "submission_forecast_T0_to_T3.nc"
OUTPUT_IMAGE = OUTPUT_DIR / "submission_visualization.png"