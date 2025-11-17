#!/usr/bin/env python3
"""Builds a tiny synthetic dataset so the training pipeline can run end-to-end."""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 - registers the .rio accessor
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config

DEFAULT_START_DATE = dt.date(2019, 1, 11)
DEFAULT_NUM_DAYS = 21
DEFAULT_GRID_SIZE = 256


def _date_sequence(start: dt.date, num_days: int) -> list[dt.date]:
    return [start + dt.timedelta(days=i) for i in range(num_days)]


def _setup_directories() -> dict[str, Path]:
    data_root = config.DATA_ROOT
    paths = {
        "ice_asc": data_root / "Ice Data" / "ice asc",
        "icecon": data_root / "Ice Data" / "ICECON" / "nc",
        "glsea": config.TRAIN_GLSEA_NC_FILE,
        "hrrr": config.TRAIN_HRRR_NC_FILE,
    }

    for target in [paths["ice_asc"], paths["icecon"], paths["glsea"].parent, paths["hrrr"].parent]:
        target.mkdir(parents=True, exist_ok=True)

    return paths


def _write_ascii(filepath: Path, values: np.ndarray) -> None:
    header = [
        f"ncols {values.shape[1]}",
        f"nrows {values.shape[0]}",
        "xllcorner 0",
        "yllcorner 0",
        "cellsize 1",
        f"nodata_value {config.ICE_ASC_NODATA_VAL}",
        "dummy 0",
    ]
    with filepath.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(header) + "\n")
        np.savetxt(handle, values, fmt="%.2f")


def _build_lat_lon(grid_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_coords = np.linspace(-90.0, -80.0, grid_size)
    y_coords = np.linspace(40.0, 50.0, grid_size)
    lon_grid, lat_grid = np.meshgrid(x_coords, y_coords)
    return x_coords, y_coords, lon_grid, lat_grid


def _make_da(name: str, data: np.ndarray, coords: dict[str, Sequence[float]]) -> xr.DataArray:
    da = xr.DataArray(data, coords=coords, dims=tuple(coords), name=name)
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    da = da.rio.write_crs("EPSG:4326")
    return da


def create_dataset(grid_size: int, num_days: int, start_date: dt.date, seed: int) -> None:
    paths = _setup_directories()
    dates = _date_sequence(start_date, num_days)
    rng = np.random.default_rng(seed)
    x_coords, y_coords, lon_grid, lat_grid = _build_lat_lon(grid_size)

    # --- Ice ASC files (land mask source) ---
    for day in dates:
        values = rng.uniform(0, 100, size=(grid_size, grid_size))
        values[: grid_size // 16, :] = config.ICE_ASC_NODATA_VAL
        _write_ascii(paths["ice_asc"] / f"g{day.strftime('%Y%m%d')}.ct", values)

        iceclass = np.full(values.shape, 1, dtype=np.int16)
        iceclass[values == config.ICE_ASC_NODATA_VAL] = 0
        iceclass[values > 70] = 4
        da = _make_da("iceclass", iceclass, {"y": y_coords, "x": x_coords})
        ds = xr.Dataset({"iceclass": da, "lat": (("y", "x"), lat_grid), "lon": (("y", "x"), lon_grid)})
        icecon_name = f"icecon_dummy_{day.strftime('%Y_%m_%d')}.nc"
        ds.to_netcdf(paths["icecon"] / icecon_name)

    # --- GLSEA water temperatures ---
    temp_stack = []
    for idx, day in enumerate(dates):
        baseline = 2.0 + 0.1 * idx
        temp_stack.append(baseline + rng.normal(0, 0.5, size=(grid_size, grid_size)))

    temp_da = xr.DataArray(
        temp_stack,
        coords={"time": pd.to_datetime(dates), "y": y_coords, "x": x_coords},
        dims=("time", "y", "x"),
        name="temp",
    )
    temp_da = temp_da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    temp_da = temp_da.rio.write_crs("EPSG:4326")
    glsea_ds = xr.Dataset({"temp": temp_da, "lat": (("y", "x"), lat_grid), "lon": (("y", "x"), lon_grid)})
    if paths["glsea"].exists():
        paths["glsea"].unlink()
    glsea_ds.to_netcdf(paths["glsea"])

    # --- HRRR weather variables ---
    weather_data = {}
    for var in ["air_temp", "windu", "windv", "PRATE_surface"]:
        stack = []
        for idx in range(len(dates)):
            baseline = idx * 0.05
            stack.append(baseline + rng.normal(0, 1, size=(grid_size, grid_size)))
        da = xr.DataArray(
            stack,
            coords={"time": pd.to_datetime(dates), "y": y_coords, "x": x_coords},
            dims=("time", "y", "x"),
            name=var,
        )
        da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
        da = da.rio.write_crs("EPSG:4326")
        weather_data[var] = da

    weather_ds = xr.Dataset(weather_data)
    weather_ds = weather_ds.assign_coords(lat=(("y", "x"), lat_grid), lon=(("y", "x"), lon_grid))
    if paths["hrrr"].exists():
        paths["hrrr"].unlink()
    weather_ds.to_netcdf(paths["hrrr"])

    print(f"Synthetic data written under {config.DATA_ROOT}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE, help="Spatial resolution to generate (default: 256)")
    parser.add_argument("--num-days", type=int, default=DEFAULT_NUM_DAYS, help="Number of sequential days to synthesize")
    parser.add_argument(
        "--start-date",
        type=lambda value: dt.datetime.strptime(value, "%Y-%m-%d").date(),
        default=DEFAULT_START_DATE,
        help="Start date for the synthetic time series (YYYY-MM-DD)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    create_dataset(arguments.grid_size, arguments.num_days, arguments.start_date, arguments.seed)
