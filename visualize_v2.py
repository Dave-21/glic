import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm
import numpy as np
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import config
import utilities 

def plot_forecast_performance(forecast_da, output_path):
    """
    Creates a comprehensive 2x4 plot showing:
    1. The 4-day forecast maps.
    2. Histograms of ice concentration for each day.
    Also prints summary statistics to the console.
    """
    print("--- Generating Enhanced Forecast Performance Report ---")
    
    # --- 1. Define Colormap and Normalization ---
    # Use a perceptually uniform colormap
    cmap = plt.cm.get_cmap('YlGnBu', 11) # 11 discrete bins
    
    # Define the bins for ice concentration
    bounds = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Set the colormap to color -1 (land) as "grey"
    cmap.set_under('grey')
    cmap.set_bad('white') # Force NaN values to be white
    
    # --- 2. Create Figure ---
    fig = plt.figure(figsize=(22, 10))
    
    # Create a 2x4 grid specification
    gs = fig.add_gridspec(2, 4)
    
    map_axes = [
        fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[0, 3], projection=ccrs.PlateCarree())
    ]
    
    hist_axes = [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[1, 3])
    ]

    # --- 3. Plot Maps and Histograms ---
    print("\n--- Summary Statistics ---")
    
    # Get total number of water pixels (where not land)
    # This is constant for all days
    total_water_pixels = float(np.count_nonzero(forecast_da.values[0] != -1))
    
    for i in range(4):
        daily_forecast = forecast_da.isel(day=i, channel=0)
        day_str = str(daily_forecast['day'].values)
        
        # --- Plot Map ---
        ax_map = map_axes[i]
        im = daily_forecast.plot.imshow(
            ax=ax_map,
            cmap=cmap,
            norm=norm, # Use the bounded normalization
            add_colorbar=False,
            transform=ccrs.PlateCarree(),
        )
        
        ax_map.add_feature(cfeature.COASTLINE.with_scale('10m'), color='black', linewidth=0.5)
        ax_map.add_feature(cfeature.BORDERS.with_scale('10m'), color='black', linewidth=0.5)
        ax_map.set_title(f"Map: {day_str}")
        ax_map.set_extent([-93, -75, 41, 49.5], crs=ccrs.PlateCarree())
        
        # --- Plot Histogram ---
        ax_hist = hist_axes[i]
        
        # Get only water/ice pixels (ignore land -1)
        water_ice_data = daily_forecast.values[daily_forecast.values != -1]
        
        ax_hist.hist(water_ice_data, bins=bounds, color='#0868ac', edgecolor='white')
        ax_hist.set_title(f"Histogram: {day_str}")
        ax_hist.set_xlabel("Ice Concentration (%)")
        ax_hist.set_ylabel("Pixel Count")
        ax_hist.set_xlim(0, 100)
        ax_hist.grid(axis='y', linestyle='--', alpha=0.7)
        
        # --- Calculate and Print Stats ---
        ice_pixels = float(np.count_nonzero(water_ice_data > 5)) # Pixels with >5% ice
        open_water_pixels = float(np.count_nonzero(water_ice_data <= 5))
        
        total_coverage_percent = (ice_pixels / total_water_pixels) * 100.0
        
        if ice_pixels > 0:
            mean_conc_on_ice = np.mean(water_ice_data[water_ice_data > 5])
        else:
            mean_conc_on_ice = 0.0
            
        print(f"\nStats for: {day_str}")
        print(f"  Total Ice Coverage (>5%): {total_coverage_percent: .2f}%")
        print(f"  Mean Concentration (on ice): {mean_conc_on_ice: .2f}%")
        print(f"  Open Water Pixels (<=5%): {int(open_water_pixels)}")

    # --- 4. Finalize Plot ---
    fig.suptitle(f"4-Day Forecast Performance Report (Model: {config.MODEL_PATH.name})", fontsize=20, y=1.04)
    
    # Add a single colorbar for all maps
    cbar = fig.colorbar(im, ax=map_axes, orientation='horizontal', fraction=0.04, pad=0.08, aspect=40)
    cbar.set_label("Ice Concentration (%)")
    cbar.set_ticks(bounds)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"\n--- Visualization Complete ---")
    print(f"Performance report saved to: {output_path}")


if __name__ == "__main__":
    if not os.path.exists(config.FORECAST_FILE):
        print(f"Error: Forecast file not found at {config.FORECAST_FILE}")
        print("Please run 'python ./run_test_forecast.py' first.")
    else:
        print(f"Loading submission forecast from {config.FORECAST_FILE}...")
        submission_ds = xr.open_dataset(config.FORECAST_FILE)
        
        # CRITICAL: Force the data to be sorted by 'y' (South-to-North)
        # This ensures cartopy plots it right-side up.
        forecast_da = submission_ds['ice_concentration_forecast'].sortby('y')

        plot_forecast_performance(forecast_da, config.OUTPUT_IMAGE)