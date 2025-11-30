import os
import sys
import urllib.request
import zipfile
import glob
from pathlib import Path
from tqdm import tqdm
import ssl

# This line bypasses SSL certificate verification.
# To fix [SSL: CERTIFICATE_VERIFY_FAILED] errors.
ssl._create_default_https_context = ssl._create_unverified_context

# --- Configuration ---

# This is the directory structure required by your config.py
# We will create all of these.
DIRECTORIES_TO_CREATE = [
    "checkpoints",
    "forecasts",
    "datasets/Shipping_Routes",                 # For downloaded shipping lanes
    "datasets/Test Data/Ice & Water Surface Temperature Initial Conditions", # For Contest Test Data
    "datasets/Test Data/Weather Data",                                       # For Contest Test Data
    "datasets/Water Surface Temperature Data/GLSEA_ICE" # For downloaded GLSEA files
]

# Publicly downloadable data
DATA_TO_DOWNLOAD = {
    "glsea_2018": {
        "url": "https://www.glerl.noaa.gov/emf/data/yyyy_glsea_ice/2018_glsea_ice.nc",
        "out_path": "datasets/Water Surface Temperature Data/GLSEA_ICE/2018_glsea_ice.nc"
    },
    "glsea_2019": {
        "url": "https://www.glerl.noaa.gov/emf/data/yyyy_glsea_ice/2019_glsea_ice.nc",
        "out_path": "datasets/Water Surface Temperature Data/GLSEA_ICE/2019_glsea_ice.nc"
    },
    "shipping_routes": {
        "url": "https://www.npms.phmsa.dot.gov/Data/CNW_V6_2024.zip",
        "out_path": "datasets/shipping_routes_temp.zip",
        "unzip_dir": "datasets/Shipping_Routes",
        "final_name": "shippinglanes"
    }
}

# --- ANSI Color Codes ---
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_step(message):
    print(f"\n{bcolors.HEADER}{bcolors.BOLD}--- {message} ---{bcolors.ENDC}")

def print_info(message):
    print(f"{bcolors.OKBLUE}[INFO]{bcolors.ENDC} {message}")

def print_success(message):
    print(f"{bcolors.OKGREEN}[SUCCESS]{bcolors.ENDC} {message}")

def print_warning(message):
    print(f"{bcolors.WARNING}[WARNING]{bcolors.ENDC} {message}")

def print_error(message):
    print(f"{bcolors.FAIL}[ERROR]{bcolors.ENDC} {message}")
    sys.exit(1)

# --- TQDM Progress Bar for Downloads ---
class TqdmUpTo(tqdm):
    """Provides `update_to(block_num, block_size, total_size)` hook for urlretrieve."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, outfile, desc=None):
    """
    Downloads a file with a TQDM progress bar.
    **Checks if file already exists and skips if it does.**
    """
    outfile = Path(outfile)
    desc = desc or outfile.name
    
    # Check if the file already exists
    if outfile.exists():
        print_info(f"Found existing file: ./{outfile}")
        print_success(f"{desc} is already downloaded.")
        return  # Exit the function, skipping the download

    print_info(f"Downloading {desc}")
    print_info(f"From: {url}")
    
    try:
        with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=desc) as t:
            urllib.request.urlretrieve(url, filename=outfile, reporthook=t.update_to)
        print_success(f"Downloaded {desc}")
    except Exception as e:
        print_error(f"Failed to download {url}. Error: {e}")

# --- Setup Functions ---

def create_directories():
    """Creates the project directory structure."""
    print_step("STEP 1: CREATING PROJECT DIRECTORIES")
    for d in DIRECTORIES_TO_CREATE:
        Path(d).mkdir(parents=True, exist_ok=True)
        print_info(f"Created/Verified: ./{d}")
    print_success("All project directories are ready.")

def show_conda_instructions():
    """Prints the conda environment creation command."""
    print_step("STEP 2: ENVIRONMENT SETUP")
    print_info("This script will not create the environment for you.")
    print_info(f"Please run the following command in your terminal to create the '{bcolors.BOLD}glic_env{bcolors.ENDC}' environment:")
    print(f"\n{bcolors.OKCYAN}conda env create -f environment.yml{bcolors.ENDC}\n")
    print_info(f"After creation, activate it with: {bcolors.BOLD}conda activate glic_env{bcolors.ENDC}")

def show_manual_data_instructions():
    """Prints instructions for manually copying contest data."""
    print_step("STEP 3: MANUAL DATA SETUP (FROM CONTEST DRIVE)")
    print_warning("The following data MUST be copied manually from the Contest Drive folder.")
    
    print(f"\n{bcolors.BOLD}1. Final Test Data (for Forecasting):{bcolors.ENDC}")
    print_info(f"Find the '{bcolors.OKCYAN}Test Data/{bcolors.ENDC}' folder on the drive.")
    print_info(f"Copy its {bcolors.BOLD}entire contents{bcolors.ENDC} (including subfolders) into this project directory:")
    print(f"   {bcolors.OKBLUE}./datasets/Test Data/{bcolors.ENDC}")
    print_info(f"Your final structure should look like this:")
    print_info(f"   ./datasets/Test Data/")
    print_info(f"   ├── Ice & Water Surface Temperature Initial Conditions/")
    print_info(f"   │   └── glsea_ice_test_initial_condition.nc")
    print_info(f"   │   └── ... (etc.)")
    print_info(f"   └── Weather Data/")
    print_info(f"       └── hrrr_weather_test_period.nc")
    print_info(f"       └── ... (etc.)")
    print_info(f"(This path is set in config.py: {bcolors.WARNING}TEST_DATA_DIR{bcolors.ENDC})")
    print(f"\n{bcolors.WARNING}NOTE ON PYTORCH:{bcolors.ENDC}")
    print_info("If PyTorch can't install when setting up env:")
    print_info(f"I installed with: {bcolors.BOLD}torch==2.9.1+cu130 torchvision==0.24.1+cu130{bcolors.ENDC}")
    print_info(f"Visit {bcolors.BOLD}https://pytorch.org/get-started/locally/{bcolors.ENDC} for the command specific to your GPU.")

    print_success("Manual data instructions complete.")

def download_public_data():
    """
    Downloads and processes all public data.
    **Skips any files or processed data that are already present.**
    """
    print_step("STEP 4: DOWNLOADING PUBLIC DATA")

    print_warning(f"If you're on a Linux and any of the below downloads hang, try: {bcolors.OKCYAN}sudo sysctl -w net.ipv4.tcp_mtu_probing=1{bcolors.ENDC}")
    
    # 1. Download GLSEA Data
    print_info("--- Checking GLSEA Data ---")
    glsea_18 = DATA_TO_DOWNLOAD['glsea_2018']
    glsea_19 = DATA_TO_DOWNLOAD['glsea_2019']
    
    # These calls will now use the new, smarter download_file function
    download_file(glsea_18['url'], glsea_18['out_path'], desc="GLSEA 2018")
    download_file(glsea_19['url'], glsea_19['out_path'], desc="GLSEA 2019")
    print_success("GLSEA data check complete.")

    # 2. Download and Process Shipping Routes
    print_info("\n--- Checking Shipping Routes ---")
    shipping_data = DATA_TO_DOWNLOAD['shipping_routes']
    zip_path = Path(shipping_data['out_path'])
    unzip_dir = Path(shipping_data['unzip_dir'])
    final_name = shipping_data['final_name']
    
    # --- MODIFICATION ---
    # Check for the *final* output file, not the temporary zip
    final_shp_file = unzip_dir / f"{final_name}.shp"
    
    if final_shp_file.exists():
        print_info(f"Found existing processed file: ./{final_shp_file}")
        print_success("Shipping routes are already processed.")
    else:
        # --- Run original logic if final file is missing ---
        print_info(f"Final file ./{final_shp_file} not found. Starting process...")
        
        # 1. Download (this will also check for the zip file)
        download_file(shipping_data['url'], zip_path, desc="Shipping Routes")
        
        # 2. Unzip
        try:
            print_info(f"Unzipping {zip_path} to {unzip_dir}...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(unzip_dir)
        except Exception as e:
            # If zip is corrupted or bad, delete it so user can re-run
            if zip_path.exists():
                zip_path.unlink()
                print_warning(f"Deleted corrupted zip {zip_path}. Please re-run the script.")
            print_error(f"Failed to unzip {zip_path}. Error: {e}")
        
        # 3. Rename
        try:
            print_info(f"Renaming shapefiles in {unzip_dir} to '{final_name}.*'.")
            shp_files = list(unzip_dir.glob("*.shp"))
            if not shp_files:
                print_error(f"No .shp file found in {unzip_dir}. Cannot rename.")
                
            original_shp = shp_files[0]
            original_basename = original_shp.stem
            
            if original_basename == final_name:
                print_info("Files are already named correctly.")
            else:
                print_info(f"Found original basename: '{original_basename}'")
                for f in unzip_dir.glob(f"{original_basename}.*"):
                    new_filename = f.name.replace(original_basename, final_name)
                    f.rename(unzip_dir / new_filename)
                print_success(f"Renamed all '{original_basename}.*' files to '{final_name}.*'.")
                
        except Exception as e:
            print_error(f"Failed to rename shapefiles. Error: {e}")
            
        # 4. Clean up zip file
        try:
            zip_path.unlink()
        except OSError as e:
            print_warning(f"Could not delete temp file {zip_path}. You can delete it manually. Error: {e}")
            
        print_success("Shipping routes processed.")

def show_runtime_data_notes():
    """Explains which data is generated or streamed, not downloaded."""
    print_step("STEP 5: NOTES ON RUNTIME DATA (NOT DOWNLOADED)")
    
    print_info(f"{bcolors.BOLD}1. HRRR Weather Data:{bcolors.ENDC}")
    print_info(f"   This data is {bcolors.BOLD}streamed live{bcolors.ENDC} from an Amazon S3 bucket (s3://hrrrzarr).")
    print_info(f"   It is {bcolors.BOLD}NOT{bcolors.ENDC} downloaded by this script. An internet connection is required to train.")

    print(f"\n{bcolors.BOLD}2. GEBCO Bathymetry Data:{bcolors.ENDC}")
    print_info(f"   The file '{bcolors.OKBLUE}./datasets/Water Surface Temperature Data/gebco_great_lakes.tif{bcolors.ENDC}'")
    print_info(f"   will be {bcolors.BOLD}auto-generated{bcolors.ENDC} by the {bcolors.WARNING}bathyreq{bcolors.ENDC} library")
    print_info(f"   the {bcolors.BOLD}first time{bcolors.ENDC} you run {bcolors.WARNING}python train.py{bcolors.ENDC}.")

    print(f"\n{bcolors.BOLD}3. Cache & Stats Files:{bcolors.ENDC}")
    print_info(f"   Files like '{bcolors.OKBLUE}weather_stats.json{bcolors.ENDC}' and '{bcolors.OKBLUE}valid_patches_*.json{bcolors.ENDC}'")
    print_info(f"   will also be {bcolors.BOLD}auto-generated{bcolors.ENDC} when you run {bcolors.WARNING}python train.py{bcolors.ENDC}.")
    print_success("Runtime data notes complete.")

def main():
    print(f"{bcolors.HEADER}====================================================={bcolors.ENDC}")
    print(f"{bcolors.HEADER}   GLIC Project Setup Script   {bcolors.ENDC}")
    print(f"{bcolors.HEADER}====================================================={bcolors.ENDC}")

    # Check if script is in project root
    if not Path("config.py").exists() or not Path("environment.yml").exists():
        print_error("This script must be run from the project root directory")
        
    create_directories()
    show_conda_instructions()
    show_manual_data_instructions()
    download_public_data()
    show_runtime_data_notes()

    print(f"\n{bcolors.HEADER}====================================================={bcolors.ENDC}")
    print(f"{bcolors.OKGREEN}{bcolors.BOLD}           SETUP SCRIPT COMPLETE!            {bcolors.ENDC}")
    print(f"{bcolors.HEADER}====================================================={bcolors.ENDC}")
    print(f"Once you have completed {bcolors.WARNING}{bcolors.BOLD}STEP 2 (Conda){bcolors.ENDC} and {bcolors.WARNING}{bcolors.BOLD}STEP 3 (Manual Data){bcolors.ENDC},")
    print("you will be ready to train the model.\n")
    print(f"1. Please run the following command in your terminal to create the '{bcolors.BOLD}glic_env{bcolors.ENDC}' environment:")
    print_info(f"\n{bcolors.OKCYAN}conda env create -f environment.yml{bcolors.ENDC}\n")
    print(f"2. {bcolors.BOLD}conda activate glic_env{bcolors.ENDC}")
    print(f"3. {bcolors.BOLD}python train.py{bcolors.ENDC}")
    print(f"4. {bcolors.BOLD}Download Test Date from CONTEST DRIVE as stated above{bcolors.ENDC}")
    print(f"5. {bcolors.BOLD}python run_test_forecast.py{bcolors.ENDC}")
    print(f"{bcolors.HEADER}====================================================={bcolors.ENDC}")

if __name__ == "__main__":

    main()
