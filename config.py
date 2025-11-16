import os
from pathlib import Path

# --- Project Root ---
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = PROJECT_ROOT / "datasets"

# --- Training Data (STILL NEEDED for Land Mask) ---
TRAIN_ICE_ASC_DIR = DATA_ROOT / "Ice Data" / "ice asc"
TRAIN_ICECON_NC_DIR = DATA_ROOT / "Ice Data" / "ICECON" / "nc"
TRAIN_GLSEA_NC_FILE = DATA_ROOT / "Water Surface Temperature Data" / "netcdf" / "glsea_20190111-20190131.nc"
TRAIN_HRRR_NC_FILE = DATA_ROOT / "Weather Data" / "High Resolution Rapid Refresh (HRRR)" / "hrrr_20190111-20190131.nc"

# --- Master Grid Definition (from training data) ---
MASTER_GRID_TEMPLATE_FILE = TRAIN_GLSEA_NC_FILE
MASTER_GRID_CRS = "EPSG:4326"  # Our target grid (standard Lat/Lon)
ICE_ASC_NATIVE_CRS = "EPSG:3175" # NAD83 / Great Lakes Albers (projected, in meters)

# --- Training Constants ---
ICE_ASC_HEADER_LINES = 7
ICE_ASC_NODATA_VAL = -1

# --- NEW: Test Data Paths ---
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