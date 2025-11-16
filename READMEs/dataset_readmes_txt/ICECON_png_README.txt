Data Description
This dataset contains Ice Cover Classification (ICECON) PNG image files. Each file represents a color-coded map of Great Lakes ice conditions derived from Synthetic Aperture Radar (SAR) imagery. The colors correspond to specific ice types and thickness categories, as described in the classification table below.
Category	Description	Example Ice Types	Thickness	Color	Code
0	Calm Water (or below noise floor)	Open Water	0?	Blue	1
1	New Lake Ice	< 2?	Green		21
2	Pancake Ice	2? – 6?	Yellow		12
3	Consolidated Flows	6? – 12?	Orange		27
4	Lake Ice with patchy crusted snow (Snow/Ice/Lake Ice)	Up to 28?	Orange		27
5	Brash Ice	> 28?, up to 9–11 m	Red		14
Land	Non-water area	—	—	Grey	1
Note: ICECON data files are not available for every day.
File Naming Convention
Each ICECON image file name follows the structure below:

Example Filenames:
RCM3_SHUB_2024_02_03_12_16_21_0760277781_091.50W_47.37N_HH_C5_GFS05CDF_glice_classes.png
S1A_ESA_2024_02_01_23_24_37_0760145077_082.02W_41.12N_VV_C5_GFS05CDF_glice_classes.png

Component Descriptions:
Component	Description
RCM3 / S1A	Platform name — RCM1, RCM2, RCM3 (RADARSAT Constellation Mission) or S1A (Sentinel?1A)
SHUB / ESA	Data source agency — SHUB (Canadian Space Agency) or ESA (European Space Agency)
2024_02_03_12_16_21	Acquisition date and time in format YYYY_MM_DD_HH_MM_SS
0760277781	Julian seconds of acquisition time
091.50W_47.37N	Image center coordinates (Longitude_W, Latitude_N)
HH / VV	Radar polarization type
C5	Radar band and GMF used for surface wind derivation
GFS05CDF	0.5° GFS wind data used as input for SAR wind correction
glice	Indicates Great Lakes ice product
classes.png	ICECON color classification image file
Data Source
This ICECON product was developed by the NOAA CoastWatch Application Team in collaboration with data provided by the US National Ice Center (NIC), Canadian Space Agency (CSA) and the European Space Agency (ESA).