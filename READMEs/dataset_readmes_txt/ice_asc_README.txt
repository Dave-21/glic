Dataset Description
This dataset contains gridded text files representing Great Lakes ice concentration and land information.
Each file consists of a 7-line header followed by a 1024 × 1024 grid of numerical values.
The files can be opened with almost any programming language (Python, R, Fortran, etc). ArcGIS is also capable of reading these files.  
Data Format
Header: 7 lines containing metadata (e.g. ncols, nrows, xllcorner, yllcorner, cellsize, etc.)
Data Section: 1024 rows × 1024 columns of integer values
Value Definitions
Value	Meaning
-1	Land
0	Water (no ice)
5, 10, …, 100	Ice concentration (%)
Spatial Resolution
Each grid cell represents an area of approximately 1800 meters (1.8 km) per pixel.
File Naming Convention
Files follow the format:
gyyyymmdd.ct
Where:
g = Great Lakes dataset identifier
yyyy = Year (4 digits)
mm = Month (2 digits)
dd = Day (2 digits)
Example:
g20190114.ct
represents data from January 14, 2019.
Data Source
The images were developed by the NOAA Great Lakes Environmental Research Laboratory (GLERL) based on Great Lakes ice concentration data from US National Ice Center (NIC).