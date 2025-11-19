# **Great Lakes IceCaster** (GLIC)

A deep learning pipeline for 3-day Great Lakes ice concentration forecasting. This system ingests GLSEA ice/temp grids, HRRR weather drivers, and GEBCO bathymetry to produce pixel-wise forecasts and automated operational narratives for the USCG.

## **Repository Structure**

* **model.py**: Custom 2D U-Net architecture with 9 input channels.  
* **train.py**: Training loop with mixed-precision (amp) and masked loss (ignores land pixels).  
* **dataset.py**: PyTorch Dataset implementation with caching, normalization, and biased sampling (prioritizing ice/shipping lanes).  
* **data\_loaders.py**: Geospatial data handling (xarray/rioxarray) for GLSEA, HRRR, and GEBCO.  
* **utilities.py**: Helper functions for master grid definition, land masking, and shipping route rasterization.  
* **visualize\_forecast\_submission.py**: The main inference engine. Generates the submission NetCDF, dashboard images, and auto-generated narrative.  
* **visualize\_forecast\_debug.py**: Tool for debugging individual model predictions and layers.  
* **config.py**: Central configuration for file paths, grid CRS (EPSG:4326), and model hyperparameters.  
* **setup.py**: Automated script to download public datasets and configure directory structure.

## **Setup**

1. Environment  
   Create the conda environment:  
   conda env create \-f environment.yml  
   conda activate glic\_env

2. Data Initialization  
   Run the setup script to create directories and download public data (GLSEA, Shipping Routes):  
   python setup.py

   *Note: You must manually place the contest "Test Data" into the datasets/ folder as prompted by the script.*

## **Usage**

### **1\. Training**

Train the U-Net model. This will generate a checkpoint in checkpoints/best\_model.pth and calculate weather statistics (weather\_stats.json).

python train.py

### **2\. Generate Forecast & Submission**

Run the full inference pipeline using the **Test Data**. This script:

* Loads the specific test files (T0 Initial Conditions \+ Weather).  
* Runs the model for T+1, T+2, and T+3.  
* Generates the CF-Compliant NetCDF (forecasts/submission\_forecast.nc).  
* Performs auto-analysis on 30+ strategic regions to generate the narrative.  
* Outputs map visualizations.

python visualize\_forecast\_submission.py

### **3\. View Dashboard**

Open docs/index.html in any web browser to view the interactive, data-driven Commander's Dashboard.

## **Methodology**

* **Input:** 9-Channel Tensor (Ice T0, Ice Delta, HRRR \[Temp, Wind U/V, Precip\], Water Temp, Bathymetry, Shipping Mask).  
* **Architecture:** 2D U-Net with 64 filters, GroupNorm, and residual connections.  
* **Loss:** Masked MSE (calculated only on water pixels).  
* **Analysis:** The system dynamically scans pixel arrays for specific ROIs (Whitefish Bay, Mackinac, etc.) to trigger operational warnings in the narrative.

## **License**

MIT License