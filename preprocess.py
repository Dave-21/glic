import utilities
import dataset
import multiprocessing
import threading
import psutil
import os
import time
import re
from functools import partial
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg') # Fix for multiprocessing: use non-interactive backend
import matplotlib.pyplot as plt
import torch
import pandas as pd
import datetime
import data_loaders
from rasterio import features
import xarray as xr
import config
import numpy as np
import geopandas as gpd

# Global dataset object for multiprocessing
# This is memory-efficient on systems with fork(), as the dataset is loaded once
# in the parent and shared with child processes via copy-on-write.
ds = None


def init_worker(dataset_instance):
    """Initializer for each worker process."""
    global ds
    ds = dataset_instance


def save_batch_worker(batch_indices, save_dir, batch_size):
    """
    Worker function to get a batch of samples from the global dataset and save them.
    """
    global ds
    try:
        batch = [ds[i] for i in batch_indices]
        if not batch:
            return "Empty batch generated."

        # Get the date from the first sample to use in the filename
        first_sample_date = batch[0].get("date")
        if not first_sample_date:
            return f"Error: No date found in first sample of batch starting at index {batch_indices[0]}."

        batch_num = batch_indices[0] // batch_size
        # Format: batch_YYYY-MM-DD_0000.pt
        filename = f"batch_{first_sample_date}_{batch_num:04d}.pt"
        save_path = save_dir / filename

        if not save_path.exists():
            torch.save(batch, save_path)
        return None
    except Exception as e:
        # Return error to be handled in the main process
        return f"Error processing batch starting at index {batch_indices[0]}: {e}"


def process_nic_shapefiles():
    """
    Finds, processes, and rasterizes NIC shapefiles into daily gridded NetCDF files.
    """
    print("--- STEP 1: Processing NIC Shapefiles -> Gridded NetCDF ---")

    shp_root_dir = config.TRAIN_NIC_SHP_DIR
    output_dir = config.NIC_PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if not shp_root_dir.exists():
        print(f"Error: Source NIC shapefile directory not found: {shp_root_dir}")
        print("Please run 'python setup.py' to download the data.")
        return False

    all_files = sorted(list(shp_root_dir.rglob("*.shp")))
    print(f"Found {len(all_files)} shapefiles to process.")

    try:
        master_grid = utilities.get_master_grid()
        height, width = master_grid.shape
        transform = master_grid.rio.transform()
    except Exception as e:
        print(f"FATAL: Could not load master grid. {e}")
        return False

    for shp_path in tqdm(all_files, desc="Rasterizing NIC Data"):
        try:
            match = re.search(r"gl(\d{6})", shp_path.name, re.IGNORECASE)
            if not match:
                continue

            dt = pd.to_datetime(match.group(1), format="%y%m%d")

            # Skip shapefiles outside the training window (+/- buffer)
            if dt.date() < (config.START_DATE - datetime.timedelta(days=1)) \
               or dt.date() > (config.END_DATE + datetime.timedelta(days=3)):
                continue

            out_name = f"NIC_{dt.strftime('%Y-%m-%d')}.nc"
            out_path = output_dir / out_name

            if out_path.exists():
                continue

            gdf = gpd.read_file(shp_path)
            if gdf.crs != master_grid.rio.crs:
                gdf = gdf.to_crs(master_grid.rio.crs)

            gdf["total_concentration"] = gdf.apply(
                lambda r: utilities.get_conc_fraction(r.get("CT")), axis=1
            )
            gdf["weighted_thickness"] = gdf.apply(
                utilities.calculate_weighted_thickness, axis=1
            )
            
            # --- Floe Size (FA) ---
            floe_col = "FA" if "FA" in gdf.columns else "F_ICE" # Fallback
            if floe_col in gdf.columns:
                gdf["floe_size"] = gdf.apply(
                    lambda r: utilities.get_floe_size_val(r.get(floe_col)), axis=1
                )
            else:
                gdf["floe_size"] = 0.0

            conc_shapes = (
                (g, v) for g, v in zip(gdf.geometry, gdf["total_concentration"])
            )
            thick_shapes = (
                (g, v) for g, v in zip(gdf.geometry, gdf["weighted_thickness"])
            )
            floe_shapes = (
                (g, v) for g, v in zip(gdf.geometry, gdf["floe_size"])
            )
            # Edge Mask: Rasterize boundaries with value 1.0
            edge_shapes = (
                (g.boundary, 1.0) for g in gdf.geometry
            )

            conc_arr = features.rasterize(
                shapes=conc_shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0.0,
                dtype=np.float32,
            )
            thick_arr = features.rasterize(
                shapes=thick_shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0.0,
                dtype=np.float32,
            )
            floe_arr = features.rasterize(
                shapes=floe_shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0.0,
                dtype=np.float32,
            )
            edge_arr = features.rasterize(
                shapes=edge_shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0.0,
                dtype=np.float32,
            )

            ds_grid = xr.Dataset(
                data_vars={
                    "ice_concentration": (("y", "x"), conc_arr),
                    "ice_thickness": (("y", "x"), thick_arr),
                    "floe_size": (("y", "x"), floe_arr),
                    "edge_mask": (("y", "x"), edge_arr),
                },
                coords=master_grid.coords,
                attrs={"source": f"NIC Shapefile: {shp_path.name}"},
            )
            ds_grid.to_netcdf(out_path)

            if config.DEBUG_MODE:
                config.DEBUG_VIZ_DIR.mkdir(exist_ok=True)
                fig, axes = plt.subplots(1, 2, figsize=(20, 10))
                ds_grid["ice_concentration"].plot(ax=axes[0])
                axes[0].set_title("Ice Concentration")
                ds_grid["ice_thickness"].plot(ax=axes[1])
                axes[1].set_title("Ice Thickness")
                plt.savefig(config.DEBUG_VIZ_DIR / f"nic_rasterized_{dt.strftime('%Y-%m-%d')}.png")
                plt.close()
        except Exception as e:
            print(f"Error processing {shp_path.name}: {e}")

    print("--- NIC Shapefile Processing Complete ---")
    return True


def compute_cfdd():
    """
    Computes Cumulative Freezing Degree Days (CFDD) from HRRR data and saves to NetCDF.
    """
    print("\n--- STEP 1.5: Computing CFDD for Training ---")
    
    # 1. Load Consolidated HRRR Data
    ds = data_loaders.get_consolidated_hrrr_dataset()
    if ds is None:
        print("Failed to load HRRR dataset.")
        return False

    print(f"Full HRRR Dataset Time Range: {ds.time.min().values} to {ds.time.max().values}")
    
    # 2. Extract Temperature
    # Find the temperature variable
    temp_var = None
    for v in ds.data_vars:
        if "air_temp" in v or "TMP" in v:
            temp_var = v
            break
            
    if temp_var is None:
        print("Could not find temperature variable.")
        return False
        
    print(f"Using temperature variable: {temp_var}")
    temps = ds[temp_var] # Kelvin
    
    # 3. Resample to Daily Mean
    print("Resampling to daily mean...")
    daily_temps = temps.resample(time="1D").mean()
    
    # 4. Compute Daily FDD
    # FDD = max(0, 273.15 - T_daily)
    freezing_point = 273.15
    degrees_below_freezing = freezing_point - daily_temps
    fdd_daily = degrees_below_freezing.clip(min=0)
    
    # 5. Compute Cumulative FDD (CFDD)
    print("Computing Cumulative Sum (CFDD)...")
    
    # Load time index to check for gaps
    times = pd.to_datetime(fdd_daily.time.values)
    time_diffs = times.to_series().diff().dt.days
    
    # Find indices where the gap is large (e.g. > 30 days), indicating a new season
    gap_indices = np.where(time_diffs > 30)[0]
    
    # We will compute cumsum for each continuous block
    cfdd_list = []
    
    # Add start index 0
    start_indices = [0] + list(gap_indices)
    end_indices = list(gap_indices) + [len(times)]
    
    for start, end in zip(start_indices, end_indices):
        print(f"Processing block: {times[start].date()} to {times[end-1].date()}")
        block_fdd = fdd_daily.isel(time=slice(start, end))
        
        # Cumsum
        block_cfdd = block_fdd.cumsum(dim="time")
        cfdd_list.append(block_cfdd)
        
    # Concatenate blocks
    if len(cfdd_list) > 1:
        full_cfdd = xr.concat(cfdd_list, dim="time")
    else:
        full_cfdd = cfdd_list[0]
        
    # 6. Save to NetCDF
    out_file = config.DATA_ROOT / "cfdd_train.nc"
    print(f"Saving CFDD to {out_file}...")
    
    # Create a Dataset
    ds_out = full_cfdd.to_dataset(name="cfdd")
    
    # Add attributes
    ds_out.attrs["description"] = "Cumulative Freezing Degree Days (CFDD) computed from HRRR 2m Air Temp"
    ds_out.attrs["units"] = "Degree-Days (C)"
    ds_out.attrs["created"] = str(datetime.datetime.now())
    
    # Save
    ds_out = ds_out.astype(np.float32)
    ds_out.to_netcdf(out_file, engine="h5netcdf")
    
    print("--- CFDD Computation Complete ---")
    return True


def monitor_ram_usage(stop_event):
    """Monitors RAM usage in a separate thread, printing if enabled in config."""
    while not stop_event.is_set():
        if config.PRINT_RAM_USAGE:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            print(f"\nRAM Usage: {mem_info.rss / 1024 ** 2:.2f} MB", flush=True)
        time.sleep(5)


def create_training_tensors():
    """
    Initializes the GreatLakesDataset and saves each sample as a .pt file
    using multiprocessing to speed up the process.
    """
    print("\n--- STEP 2: Creating Fast-Load Training Tensors ---")

    save_dir = config.DATA_ROOT / "processed_tensors"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Start RAM monitoring
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_ram_usage, args=(stop_event,))
    monitor_thread.daemon = True
    monitor_thread.start()

    print("Initializing Master Dataset (this will load all data into memory)...")
    try:
        # Load the full dataset in the main process
        main_ds = dataset.GreatLakesDataset(is_train=True)
    except ValueError as e:
        print(f"\nFATAL ERROR during dataset initialization: {e}")
        stop_event.set()
        monitor_thread.join()
        return

    if len(main_ds) == 0:
        print("\nWarning: The dataset is empty. No tensors will be created.")
        stop_event.set()
        monitor_thread.join()
        return

    print(f"\nSaving {len(main_ds)} tensor samples to {save_dir}...")

    # Dynamic Worker Allocation
    # Ryzen 9700X has 8 cores / 16 threads. 16GB RAM.
    # We estimate ~1.5GB RAM per worker to be safe.
    import psutil
    available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
    cpu_count = os.cpu_count()
    
    # Heuristic: Min(CPUs, AvailableRAM / 3.0GB)
    # But cap at CPU count - 2 to leave room for system/main process
    max_workers_by_ram = int(available_ram_gb / 3.0)
    # num_workers = max(1, min(cpu_count - 2, max_workers_by_ram))
    
    # FORCE SINGLE THREADED TO FIX CRASH
    num_workers = 0 
    
    print(f"Dynamic Config: {available_ram_gb:.2f} GB RAM available, {cpu_count} CPUs.")
    print(f"Using {num_workers} worker processes (Single Threaded Mode).")

    batch_size = 20
    indices = range(len(main_ds))
    batches = [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]

    # Dynamic RAM-Aware Processing
    # Create a partial function with the save_dir argument fixed
    worker_func = partial(save_batch_worker, save_dir=save_dir, batch_size=batch_size)

    # We allow up to 6 workers, but throttle based on RAM usage.
    max_workers = 6
    print(f"Starting dynamic processing with up to {max_workers} workers...")
    
    # maxtasksperchild=1 ensures workers restart and free memory after every batch
    pool = multiprocessing.Pool(
        processes=max_workers, 
        initializer=init_worker, 
        initargs=(main_ds,),
        maxtasksperchild=1
    )
    
    active_futures = []
    results = []
    batch_iterator = iter(batches)
    pbar = tqdm(total=len(batches), desc="Dynamic Processing")
    
    try:
        while True:
            # 1. Check RAM
            mem_percent = psutil.virtual_memory().percent
            
            # 2. Clean up finished futures
            # Iterate backwards to safely remove
            for i in range(len(active_futures) - 1, -1, -1):
                if active_futures[i].ready():
                    # Check for exceptions
                    try:
                        res = active_futures[i].get()
                        results.append(res)
                    except Exception as e:
                        print(f"\nWorker error: {e}")
                    
                    active_futures.pop(i)
                    pbar.update(1)
            
            # 3. Check if we are done
            if len(active_futures) == 0 and batch_iterator.__length_hint__() == 0:
                # Note: __length_hint__ might not be reliable for all iterators but works for list iter
                # Better check: we need a flag if iterator is exhausted
                pass 

            # 4. Submit new tasks
            # Criteria: 
            # - Slots available in pool
            # - RAM is safe (< 85%) OR we have NO active workers (must make progress)
            # - Iterator not exhausted
            
            if len(active_futures) < max_workers:
                if mem_percent < 85.0 or len(active_futures) == 0:
                    try:
                        batch = next(batch_iterator)
                        f = pool.apply_async(worker_func, (batch,))
                        active_futures.append(f)
                    except StopIteration:
                        if len(active_futures) == 0:
                            break # All done
                        # Else: just wait for remaining
                else:
                    # RAM high, wait for tasks to finish
                    time.sleep(0.5)
            else:
                # Pool full, wait
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\nProcessing interrupted. Terminating pool...")
        pool.terminate()
        pool.join()
        sys.exit(1)
        
    pool.close()
    pool.join()
    pbar.close()

    # Stop RAM monitoring
    stop_event.set()
    monitor_thread.join()
    print("\nRAM monitoring stopped.")

    # Check for any errors returned by workers
    errors = [res for res in results if res is not None]
    if errors:
        print("\n--- Errors occurred during tensor saving ---")
        for error in errors:
            print(error)
        print(f"\n{len(errors)}/{len(batches)} batches failed to save.")
    else:
        print("\n--- Pre-processing Complete ---")
        print(f"Fast-load tensors saved to: {save_dir}")


if __name__ == "__main__":
    # Ensure multiprocessing works correctly when run as a script
    multiprocessing.set_start_method("fork", force=True)
    if process_nic_shapefiles():
        if compute_cfdd():
            create_training_tensors()