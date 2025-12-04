import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm
import ssl
import config
from datetime import timedelta, date
import xarray as xr
import data_loaders
import utilities

# This line bypasses SSL certificate verification.
ssl._create_default_https_context = ssl._create_unverified_context

# --- Configuration ---
DIRECTORIES_TO_CREATE = [
    "checkpoints",
    "forecasts",
    "datasets/Shipping_Routes",
    "datasets/Test Data/Ice & Water Surface Temperature Initial Conditions",
    "datasets/Test Data/Weather Data",
    "datasets/Water Surface Temperature Data/GLSEA_ICE",
    config.TRAIN_NIC_SHP_DIR,
    config.NIC_PROCESSED_DIR,
    config.TRAIN_NIC_SHP_DIR.parent / "temp_zips",
    config.TRAIN_HRRR_NC_FILE.parent,
]

DATA_TO_DOWNLOAD = {
    "glsea_2018": {
        "url": "https://www.glerl.noaa.gov/emf/data/yyyy_glsea_ice/2018_glsea_ice.nc",
        "out_path": "datasets/Water Surface Temperature Data/GLSEA_ICE/2018_glsea_ice.nc",
    },
    "glsea_2019": {
        "url": "https://www.glerl.noaa.gov/emf/data/yyyy_glsea_ice/2019_glsea_ice.nc",
        "out_path": "datasets/Water Surface Temperature Data/GLSEA_ICE/2019_glsea_ice.nc",
    },
    "shipping_routes": {
        "url": "https://www.npms.phmsa.dot.gov/Data/CNW_V6_2024.zip",
        "out_path": "datasets/shipping_routes_temp.zip",
        "unzip_dir": "datasets/Shipping_Routes",
        "final_name": "shippinglanes",
    },
}


class bcolors:
    HEADER, OKBLUE, OKCYAN, OKGREEN, WARNING, FAIL, ENDC, BOLD = (
        "\033[95m",
        "\033[94m",
        "\033[96m",
        "\033[92m",
        "\033[93m",
        "\033[91m",
        "\033[0m",
        "\033[1m",
    )


def print_step(msg):
    print(f"\n{bcolors.HEADER}{bcolors.BOLD}--- {msg} ---{bcolors.ENDC}")


def print_info(msg):
    print(f"{bcolors.OKBLUE}[INFO]{bcolors.ENDC} {msg}")


def print_success(msg):
    print(f"{bcolors.OKGREEN}[SUCCESS]{bcolors.ENDC} {msg}")


def print_warning(msg):
    print(f"{bcolors.WARNING}[WARNING]{bcolors.ENDC} {msg}")


def print_error(msg):
    print(f"{bcolors.FAIL}[ERROR]{bcolors.ENDC} {msg}")
    sys.exit(1)


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, outfile, desc=None, ignore_if_exists=True):
    outfile = Path(outfile)
    if ignore_if_exists and outfile.exists():
        print_info(f"Skipping existing file: ./{outfile}")
        return True
    print_info(f"Downloading {desc or outfile.name} from {url}")
    try:
        with TqdmUpTo(
            unit="B", unit_scale=True, miniters=1, desc=desc or outfile.name
        ) as t:
            urllib.request.urlretrieve(url, filename=outfile, reporthook=t.update_to)
        return True
    except Exception as e:
        print_warning(f"Failed to download {url}. Error: {e}")
        if outfile.exists():
            outfile.unlink()
        return False


def create_directories():
    print_step("STEP 1: CREATING PROJECT DIRECTORIES")
    for d in DIRECTORIES_TO_CREATE:
        Path(d).mkdir(parents=True, exist_ok=True)
        print_info(f"Created/Verified: ./{d}")
    print_success("All project directories are ready.")


def download_nic_data():
    print_step("STEP 4a: DOWNLOADING USNIC SHAPEFILES")
    target_dir, temp_zip_dir = (
        config.TRAIN_NIC_SHP_DIR,
        config.TRAIN_NIC_SHP_DIR.parent / "temp_zips",
    )
    current_date, delta = config.START_DATE, timedelta(days=1)
    while current_date <= config.END_DATE:
        if not any(target_dir.rglob(f"*{current_date.strftime('%y%m%d')}*.shp")):
            url = f"https://usicecenter.gov/File/DownloadArchive?prd=35{current_date.strftime('%m%d%Y')}"
            zip_path = temp_zip_dir / f"nic_{current_date.strftime('%Y%m%d')}.zip"
            if download_file(
                url, zip_path, desc=f"NIC Data {current_date}", ignore_if_exists=False
            ):
                try:
                    if zip_path.stat().st_size > 2000:
                        with zipfile.ZipFile(zip_path, "r") as zf:
                            zf.extractall(target_dir)
                        print_success(f"  -> Extracted {current_date}.")
                    else:
                        print_warning(f"  -> Skipping empty file for {current_date}.")
                except Exception as e:
                    print_warning(f"  -> Extraction error for {current_date}: {e}")
                if zip_path.exists():
                    zip_path.unlink()
        current_date += delta
    print_success("USNIC data download check complete.")


def download_hrrr_data():
    print_step("STEP 4b: DOWNLOADING HRRR TRAINING DATA (THIS MAY TAKE A LONG TIME)")
    if config.TRAIN_HRRR_NC_FILE.exists():
        print_success("Consolidated HRRR training file already exists.")
        return

    print_info("This is a one-time download and processing step.")
    master_grid = utilities.get_master_grid()
    all_days = [
        config.START_DATE + timedelta(days=x)
        for x in range((config.END_DATE - config.START_DATE).days + 1)
    ]
    hrrr_datasets = []

    for day in tqdm(all_days, desc="Downloading and reprojecting daily HRRR data"):
        ds = data_loaders.download_and_reproject_hrrr_for_day(day, master_grid)
        if ds:
            hrrr_datasets.append(ds)

    if not hrrr_datasets:
        print_error(
            "Failed to download any HRRR data. Cannot create consolidated file."
        )
        return

    print_info("Concatenating all daily HRRR datasets...")
    full_hrrr_ds = xr.concat(hrrr_datasets, dim="time")
    print_info(f"Saving consolidated HRRR data to {config.TRAIN_HRRR_NC_FILE}...")
    full_hrrr_ds = full_hrrr_ds.drop_vars("metpy_crs", errors="ignore")
    full_hrrr_ds.to_netcdf(config.TRAIN_HRRR_NC_FILE)
    print_success("Consolidated HRRR file created.")


def download_public_data():
    print_step("STEP 4: DOWNLOADING PUBLIC DATA")
    for key, data in DATA_TO_DOWNLOAD.items():
        if "unzip_dir" not in data:
            download_file(data["url"], data["out_path"])

    shipping_data = DATA_TO_DOWNLOAD["shipping_routes"]
    final_shp = Path(shipping_data["unzip_dir"]) / f"{shipping_data['final_name']}.shp"
    if not final_shp.exists():
        zip_path = Path(shipping_data["out_path"])
        if download_file(shipping_data["url"], zip_path, ignore_if_exists=False):
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(shipping_data["unzip_dir"])
                shp = next(Path(shipping_data["unzip_dir"]).glob("*.shp"))
                if shp.stem != shipping_data["final_name"]:
                    for f in Path(shipping_data["unzip_dir"]).glob(f"{shp.stem}.*"):
                        f.rename(
                            Path(shipping_data["unzip_dir"])
                            / f.name.replace(shp.stem, shipping_data["final_name"])
                        )
            except Exception as e:
                print_error(f"Failed processing {zip_path}: {e}")
            if zip_path.exists():
                zip_path.unlink()
    else:
        print_info("Skipping existing processed shipping routes.")


def main():
    if not Path("config.py").exists():
        print_error("Script must be run from project root.")
    create_directories()
    print_step("STEP 2: ENVIRONMENT SETUP (Manual)")
    print_info(
        "Please run 'mamba env create -f environment.yml' and 'mamba activate glic_env'"
    )
    print_step("STEP 3: MANUAL DATA SETUP (See README)")
    download_public_data()
    download_nic_data()
    download_hrrr_data()
    print_step("STEP 5: FINAL NOTES")
    print_info(
        "GEBCO bathymetry data will be auto-generated on first run of preprocess.py."
    )
    print_success("\nSETUP SCRIPT COMPLETE!")
    print(
        "Next Steps:\n1. Activate your conda environment.\n2. Run 'python preprocess.py'.\n3. Run 'python train.py'."
    )


if __name__ == "__main__":
    main()
