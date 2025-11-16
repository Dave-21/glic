Data Description
This dataset contains Ice Cover Classification (ICECON) NetCDF (.nc) files. Each file represents a Level 2 product derived from SAR imagery. The primary variable in each file is 'iceclass'. The values in 'iceclass' correspond to the codes in the following classification table:
Category	Description	Example Ice Types	Thickness	Color	Code
0	Calm Water (or below noise floor)	Open Water	0?	Blue	1
1	New Lake Ice	< 2?	Green		21
2	Pancake Ice	2? – 6?	Yellow		12
3	Consolidated Flows	6? – 12?	Orange		27
4	Lake Ice w/ patchy crusted snow (Snow/Ice/LakeIce)	Up to 28?	Orange		27
5	Brash Ice	> 28?, up to 9–11 m	Red		14
Land	Non-water area	—	—	Grey	1

Note: Categories 3 and 4 share the same color (orange) and code number (27).
CoastWatch free software cw utilites (CDAT): https://coastwatch.noaa.gov/cwn/data-access-tools/coastwatch-utilities.html#downloads
File Naming Convention
Example filenames:
RCM3_SHUB_2024_02_03_12_16_21_0760277781_091.50W_47.37N_HH_C5_GFS05CDF_glice_level2.nc
S1A_ESA_2024_02_01_23_24_37_0760145077_082.02W_41.12N_VV_C5_GFS05CDF_glice_level2.nc
Component descriptions:
Component	Description
RCM3 / S1A	Platform name (RCM1, RCM2, RCM3 for RADARSAT; S1A for Sentinel?1A)
SHUB / ESA	Data source agency (SHUB for Canadian Space Agency; ESA for European Space Agency)
2024_02_03_12_16_21	Acquisition time (YYYY_MM_DD_HH_MM_SS)
0760277781	Julian seconds of acquisition time
091.50W_47.37N	Center coordinates of the image (Longitude_W, Latitude_N)
HH / VV	Polarization type
C5	Radar band and GMF used for surface wind derivation
GFS05CDF	0.5° GFS wind data used as input for SAR wind correction
glice	Great Lakes ice product
level2	Data processing level (Level 2)


Data Source
This product was developed by the NOAA CoastWatch Application Team.