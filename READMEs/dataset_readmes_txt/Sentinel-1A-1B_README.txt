Data Description
This dataset contains Sentinel-1A/B Synthetic Aperture Radar (SAR) imagery products acquired over the Great Lakes region. The Sentinel-1 mission, operated by the European Space Agency (ESA) under the Copernicus program, provides C-band radar data for monitoring land and ocean surfaces in all weather conditions, day or night.
The data are collected in Interferometric Wide Swath (IW) mode using dual polarization (VV/VH) or single polarization (VV). The product type is Ground Range Detected (GRD), which represents radar backscatter intensity that has been detected, multi-looked, and projected to ground range using an Earth ellipsoid model. These products include measurement data in GeoTIFF format and accompanying metadata in XML format. 
Each image provides information on surface backscatter, which can be used to infer ice type, concentration, and motion over the Great Lakes.
Note: The data files are not available for every day.

File Naming Convention
Example filename: S1A_IW_GRDH_1SDV_20190127T233204_20190127T233229_025670_02D9F1_D2EC.SAFE
Component	Description
S1A / S1B	Satellite identifier (Sentinel-1A or Sentinel-1B)
IW	Acquisition mode: Interferometric Wide Swath
GRDH	Product type: Ground Range Detected, High Resolution
1SDV	Processing level (1 = Level-1), Standard product, Dual polarization (VV/VH)
20190127T233204	Start time of acquisition (UTC)
20190127T233229	Stop time of acquisition (UTC)
025670	Orbit number
02D9F1	Processing baseline (software configuration)
D2EC	Unique product identifier
.SAFE	Data container folder including metadata and image data 
Note: In some cases, the .SAFE extension may be omitted from file names.
Data Source
Sentinel-1A/B data are distributed by the European Space Agency (ESA) through the Copernicus Open Access Hub. The data files were modified by the NOAA CoastWatch Program. The data used in this project were obtained and processed by the NOAA Great Lakes CoastWatch Node and the Great Lakes Environmental Research Laboratory (GLERL).