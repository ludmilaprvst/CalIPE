.. CalIPE_doc documentation master file, created by
   sphinx-quickstart on Fri Jul  7 17:54:12 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Input data files
================
In the CalIPE package, three type of data files are possible.
The two of them, the Evt file and the Obs file, are mandatory and contains the intensity database and the metadata associated.
The third one, the Region file, describes the geographical limit of different regions.

Evt file
--------

Description of the file
"""""""""""""""""""""""

The Evt file contains the metadata of the calibration earthquakes. The Evt file is a .csv or a .txt
file, with columns separeted by ;. Each line contains the metadata associatedd to one earthquake.
The columns are:
	- EVID: ID of the earthquake,
	- Year: year of occurrrence of the earthquake,
	- Month: month of occurence of the earthquake,
	- Day: day of occurence of the earthquake,
	- Lon: longitude in WGS84 of the earthquake epicenter location,
	- Lat: latitude in WGS84 of the earthquake epicenter location,
	- QPos: quality of the epicenter location. A is the best quality and E the worst.
	- I0: epicentral intensity of the earthquake,
	- QI0: quality of the epicentral intensity value, ranked between A (very good) and E (very uncertain)
	- Ic: intensity of completeness of the earthquake macroseismic field. Intensity classes smaller than
	  Ic are not considered as complete,
	- Dc: distance of completeness of the earthquake macroseismic field. For distances smaller than Dc,
          no abrupt changes in data density with distance is observed,
	- Mag: magnitude of the earthquake. This magnitude should be the most homogeneous possible among
          the calibration dataset,  
	- StdM: uncertainty associated to the magnitude,
	- Depth: hypocentral depth of the earthquake,
	- Hmin: shallow limit of depth uncertainty,
	- Hmax: deep limit of depth uncertainty,
	- QH: Quality of the instrumental depth estimate (including the depth limits), ranked between A (very good) and E (unknown)

An example of the Evt file is available in the CalIPE_examples repository (address), in the Data folder.
When creating your own Evt file, please respect the columns names.

Comments about the needed metadata
""""""""""""""""""""""""""""""""""

The QPos parameter is not used in the calibration process. However, a column with this name is needed to prepare
the intensity data for the calibration process. 
The magnitude and its associated uncertainty are not needed for the calibration of the Koveslighety 
mathematical formulation:
	- I = I0 + beta.log10(Hypo/H) + gamma.(Hypo/H)
	- I = I0 + beta.log10(Hypo/H)

If you calibrate this mathematical formulation, you don't need to fill this column with the magnitude value,
especially if you don't know the magnitude. In this case, you can put -99 or any other values in this column.

However, to be sure that the CalIPE tool will run, please fill all the columns described in the Evt file
description.

Obs file
--------

Description of the file
"""""""""""""""""""""""
The Obs file contains the macroseismic field of the calibration earthquakes.The Obs file is a .csv or a .txt
file, with columns separeted by ;. Each line describe the intenisty value at one locality for one earthquake.
The columns are:
	- EVID: ID of the earthquake,
	- Lon: longitude in WGS84 of the locality,
	- Lat: latitude in WGS84 of the locality,
	- Iobs: value of the intensity at the locality
	- QIobs: quality of the value of intensity at the locality. Quality A stands for very certain, 
	quality B for fairly certain and C for uncertain.


An example of the Obs file is available in the CalIPE_examples repository (address), in the Data folder.
When creating your own Obs file, please respect the columns names.

The Obs file should at least contain the macroseismic field of the calibration earthquake and can contain
other macroseismic field. The earthquake ID should be the same in the Evt file and the Obs file.

Region file
-----------

The file is mandatory if a regional weighting scheme or a regional coefficient C1 estimation in the two-step strategy.
This file describe the limits of the chosen region. Each region is identified by an ID. The Region file is a .txt
file, with columns separeted by ;.
The columns are:
	- ID_region: ID of the region,
	- Lon: longitude coordinate in WGS84 of one the point decribing the region polygon,
	- Lat: latitude coordinate in WGS84 of one the point decribing the region polygon,

An example of the Region file is available in the CalIPE_examples repository (address), in the Data/Regions folder.



.. toctree::
   :maxdepth: 2
   :caption: Contents:



